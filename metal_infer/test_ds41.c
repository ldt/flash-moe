/*
 * test_ds41.c — Unit tests for ds41_ops.h (DeepSeek-V4.1-Flash CPU reference ops).
 *
 * Build & run (any platform, no Metal needed):
 *     make test-ds41
 * Optional Engram file cross-check against export_ds41_engram_meta.py:
 *     ./test_ds41 --engram ds41_engram_meta.bin 0 128803 5726 1000 2 128804
 * prints one line per (layer, position) with the 24 row indices, to diff against
 *     uv run deepseek_v41/export_ds41_engram_meta.py ... --selftest 0 128803 5726 1000 2 128804
 */
#define DS41_OPS_IMPL
#include "ds41_ops.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("  FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); } } while (0)

static uint32_t rng_state = 0x12345678u;
static uint32_t rnd(void) { rng_state ^= rng_state << 13; rng_state ^= rng_state >> 17; rng_state ^= rng_state << 5; return rng_state; }
static float frand(void) { return (float)(rnd() % 20001) / 10000.0f - 1.0f; }

static void test_scalar_decoders(void) {
    printf("[e8m0] all 255 encodings vs ldexp\n");
    for (int e = 0; e < 255; e++) CHECK(ds41_e8m0_to_f32((uint8_t)e) == ldexpf(1.0f, e - 127), "e8m0 %d", e);
    CHECK(ds41_e8m0_to_f32(127) == 1.0f && ds41_e8m0_to_f32(128) == 2.0f && ds41_e8m0_to_f32(126) == 0.5f, "e8m0 anchors");

    printf("[e4m3] all 256 encodings vs arithmetic formula\n");
    for (int v = 0; v < 256; v++) {
        int s = v >> 7, e = (v >> 3) & 15, m = v & 7;
        double want;
        if (e == 15 && m == 7) want = 0.0;                 /* NaN -> 0 by our convention */
        else if (e == 0) want = m * pow(2.0, -9);
        else want = pow(2.0, e - 7) * (1.0 + m / 8.0);
        if (s) want = -want;
        CHECK(ds41_e4m3_to_f32((uint8_t)v) == (float)want, "e4m3 0x%02x: %g vs %g", v, ds41_e4m3_to_f32((uint8_t)v), want);
    }
    CHECK(ds41_e4m3_to_f32(0x7E) == 448.0f, "e4m3 max 448");
    CHECK(ds41_e4m3_to_f32(0x38) == 1.0f, "e4m3 1.0");
    CHECK(ds41_e4m3_to_f32(0xB8) == -1.0f, "e4m3 -1.0");
    CHECK(ds41_e4m3_to_f32(0x08) == 0.015625f, "e4m3 smallest normal 2^-6");

    printf("[e2m1] table matches convert.py FP4_TABLE\n");
    const float want[16] = {0, .5f, 1, 1.5f, 2, 3, 4, 6, 0, -.5f, -1, -1.5f, -2, -3, -4, -6};
    for (int i = 0; i < 16; i++) CHECK(DS41_E2M1_LUT[i] == want[i], "e2m1 %d", i);

    CHECK(ds41_bf16_to_f32(0x3F80) == 1.0f && ds41_bf16_to_f32(0xC000) == -2.0f, "bf16");
}

static void test_mxfp4_matvec(void) {
    printf("[mxfp4] matvec vs double reference (out=64, in=256)\n");
    const int out_dim = 64, in_dim = 256;
    uint8_t *W = malloc(out_dim * in_dim / 2), *S = malloc(out_dim * in_dim / 32);
    float *x = malloc(sizeof(float) * in_dim), *y = malloc(sizeof(float) * out_dim);
    for (int i = 0; i < out_dim * in_dim / 2; i++) W[i] = (uint8_t)(rnd() & 0xFF);
    for (int i = 0; i < out_dim * in_dim / 32; i++) S[i] = (uint8_t)(120 + rnd() % 12);
    for (int i = 0; i < in_dim; i++) x[i] = frand();
    ds41_mxfp4_matvec(W, S, x, y, out_dim, in_dim);
    double max_err = 0.0;
    for (int o = 0; o < out_dim; o++) {
        double ref = 0.0;
        for (int i = 0; i < in_dim; i++) {
            uint8_t byte = W[o * in_dim / 2 + i / 2];
            int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
            double scale = ldexp(1.0, S[o * (in_dim / 32) + i / 32] - 127);
            ref += DS41_E2M1_LUT[nib] * scale * (double)x[i];
        }
        double err = fabs(ref - y[o]) / (fabs(ref) + 1e-3);
        if (err > max_err) max_err = err;
    }
    CHECK(max_err < 1e-4, "mxfp4 matvec rel err %g", max_err);
    printf("        max rel err %.2e\n", max_err);

    /* row dequant agrees with the matvec (x = one-hot) */
    float *row = malloc(sizeof(float) * in_dim);
    ds41_mxfp4_dequant_row(W + 3 * in_dim / 2, S + 3 * (in_dim / 32), row, in_dim);
    for (int i = 0; i < in_dim; i += 37) {
        memset(x, 0, sizeof(float) * in_dim); x[i] = 1.0f;
        ds41_mxfp4_matvec(W, S, x, y, out_dim, in_dim);
        CHECK(y[3] == row[i], "dequant_row col %d: %g vs %g", i, row[i], y[3]);
    }
    free(W); free(S); free(x); free(y); free(row);
}

static void test_fp8blk_matvec(void) {
    printf("[fp8blk] matvec vs double reference (out=96, in=160, 32x32 scales)\n");
    const int out_dim = 96, in_dim = 160;
    uint8_t *W = malloc(out_dim * in_dim), *S = malloc((out_dim / 32) * (in_dim / 32));
    float *x = malloc(sizeof(float) * in_dim), *y = malloc(sizeof(float) * out_dim);
    for (int i = 0; i < out_dim * in_dim; i++) { uint8_t v = (uint8_t)(rnd() & 0xFF); if ((v & 0x7F) == 0x7F) v = 0x38; W[i] = v; }
    for (int i = 0; i < (out_dim / 32) * (in_dim / 32); i++) S[i] = (uint8_t)(115 + rnd() % 20);
    for (int i = 0; i < in_dim; i++) x[i] = frand();
    ds41_fp8blk_matvec(W, S, x, y, out_dim, in_dim);
    double max_err = 0.0;
    for (int o = 0; o < out_dim; o++) {
        double ref = 0.0;
        for (int i = 0; i < in_dim; i++)
            ref += (double)ds41_e4m3_to_f32(W[o * in_dim + i]) * ldexp(1.0, S[(o / 32) * (in_dim / 32) + i / 32] - 127) * x[i];
        double err = fabs(ref - y[o]) / (fabs(ref) + 1e-3);
        if (err > max_err) max_err = err;
    }
    CHECK(max_err < 1e-4, "fp8blk matvec rel err %g", max_err);
    printf("        max rel err %.2e\n", max_err);
    free(W); free(S); free(x); free(y);
}

static void test_swiglu_and_norm(void) {
    printf("[swiglu] clamp semantics (up both sides, gate above only)\n");
    float gate[4] = {-30.0f, 0.0f, 12.0f, 2.0f}, up[4] = {-30.0f, 5.0f, 12.0f, -12.0f}, out[4];
    ds41_swiglu_clamped(gate, up, out, 4, DS41_SWIGLU_LIMIT);
    float silu_m30 = -30.0f / (1.0f + expf(30.0f));      /* gate not clamped below */
    CHECK(fabsf(out[0] - silu_m30 * -10.0f) < 1e-6f, "swiglu[0] %g", out[0]);
    CHECK(out[1] == 0.0f, "swiglu[1] %g", out[1]);
    float silu_10 = 10.0f / (1.0f + expf(-10.0f));
    CHECK(fabsf(out[2] - silu_10 * 10.0f) < 1e-4f, "swiglu[2] %g", out[2]);
    float silu_2 = 2.0f / (1.0f + expf(-2.0f));
    CHECK(fabsf(out[3] - silu_2 * -10.0f) < 1e-5f, "swiglu[3] %g", out[3]);

    printf("[rmsnorm] unit weight, eps=1e-20\n");
    float x[8], w1f[8], y[8];
    uint16_t w[8];
    for (int i = 0; i < 8; i++) { x[i] = (float)(i + 1); w[i] = 0x3F80; w1f[i] = 1.0f; }
    (void)w1f;
    ds41_rms_norm(x, w, y, 8, DS41_NORM_EPS);
    double ss = 0; for (int i = 0; i < 8; i++) ss += x[i] * x[i];
    for (int i = 0; i < 8; i++) CHECK(fabs(y[i] - x[i] / sqrt(ss / 8)) < 1e-5, "rmsnorm %d", i);
}

static void test_routing(void) {
    printf("[route] sqrtsoftplus + noaux_tc bias + norm_topk_prob + route_scale\n");
    /* 6 experts, K=3. Bias pushes expert 5 into the top-3 and expert 0 out of it. */
    float logits[6] = {3.0f, 2.5f, 2.0f, -1.0f, -2.0f, -3.0f};
    float bias[6]   = {-5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f};
    int idx[3]; float w[3];
    ds41_route(logits, bias, 6, 3, DS41_ROUTE_SCALE, idx, w);
    CHECK(idx[0] == 5 && idx[1] == 1 && idx[2] == 2, "route idx %d %d %d", idx[0], idx[1], idx[2]);
    float s1 = sqrtf(log1pf(expf(2.5f))), s2 = sqrtf(log1pf(expf(2.0f))), s5 = sqrtf(log1pf(expf(-3.0f)));
    float sum = s1 + s2 + s5;
    CHECK(fabsf(w[0] - 1.5f * s5 / sum) < 1e-6f, "route w0 %g", w[0]);
    CHECK(fabsf(w[1] - 1.5f * s1 / sum) < 1e-6f, "route w1 %g", w[1]);
    CHECK(fabsf(w[2] - 1.5f * s2 / sum) < 1e-6f, "route w2 %g", w[2]);
    CHECK(fabsf(w[0] + w[1] + w[2] - 1.5f) < 1e-5f, "route weights sum to route_scale");
    /* softplus threshold branch */
    float big[2] = {25.0f, 0.0f}; int i2[1]; float w2[1];
    ds41_route(big, NULL, 2, 1, 1.0f, i2, w2);
    /* norm_topk_prob only applies for topk > 1 (Gate.forward), so K=1 keeps the raw score sqrt(25). */
    CHECK(i2[0] == 0 && fabsf(w2[0] - 5.0f) < 1e-6f, "route K=1 keeps raw sqrtsoftplus score: %g", w2[0]);
}

static void test_hc(void) {
    printf("[hc] Sinkhorn comb is doubly stochastic, pre/post ranges, pre/post roundtrip\n");
    const int hc = DS41_HC_MULT, mix_hc = (2 + hc) * hc, dim = 16;
    float mixes[24], scale3[3] = {0.7f, 1.3f, 2.0f}, base[24];
    for (int i = 0; i < mix_hc; i++) { mixes[i] = 3.0f * frand(); base[i] = frand(); }
    float pre[4], post[4], comb[16];
    ds41_hc_split_sinkhorn(mixes, scale3, base, hc, DS41_HC_SINKHORN_IT, DS41_HC_EPS, pre, post, comb);
    for (int j = 0; j < hc; j++) {
        CHECK(pre[j] > 0.0f && pre[j] < 1.0f + 2e-6f, "pre[%d]=%g", j, pre[j]);
        CHECK(post[j] > 0.0f && post[j] < 2.0f, "post[%d]=%g", j, post[j]);
        float rs = 0, cs = 0;
        for (int k = 0; k < hc; k++) { rs += comb[j * hc + k]; cs += comb[k * hc + j]; }
        /* the last Sinkhorn step normalizes columns, so columns are exact and rows converge */
        CHECK(fabsf(rs - 1.0f) < 2e-2f, "row sum %d = %g", j, rs);
        CHECK(fabsf(cs - 1.0f) < 1e-4f, "col sum %d = %g", j, cs);
    }
    /* with many more iterations the rows must converge too (checks the alternating updates) */
    ds41_hc_split_sinkhorn(mixes, scale3, base, hc, 400, DS41_HC_EPS, pre, post, comb);
    for (int j = 0; j < hc; j++) {
        float rs = 0;
        for (int k = 0; k < hc; k++) rs += comb[j * hc + k];
        CHECK(fabsf(rs - 1.0f) < 1e-4f, "row sum after 400 it %d = %g", j, rs);
    }
    /* hc_mixes: x = ones, hc_fn rows = e_i -> mixes = rsqrt(mean(x^2)) * 1 = 1 */
    float x[64], hc_fn[24 * 64], out_mixes[24];
    for (int i = 0; i < 64; i++) x[i] = 1.0f;
    memset(hc_fn, 0, sizeof hc_fn);
    for (int m = 0; m < mix_hc; m++) hc_fn[m * 64 + m] = 1.0f;
    ds41_hc_mixes(x, hc_fn, hc, dim, DS41_NORM_EPS, out_mixes);
    for (int m = 0; m < mix_hc; m++) CHECK(fabsf(out_mixes[m] - 1.0f) < 1e-6f, "hc_mixes[%d]=%g", m, out_mixes[m]);
    /* hc_pre with one-hot pre selects copy 2; hc_post with comb=I, post=0 returns the residual */
    float pre1[4] = {0, 0, 1, 0}, y[16], res[64], post0[4] = {0, 0, 0, 0}, I[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}, back[64];
    for (int i = 0; i < 64; i++) res[i] = (float)i;
    ds41_hc_pre(res, pre1, hc, dim, y);
    for (int d = 0; d < dim; d++) CHECK(y[d] == res[2 * dim + d], "hc_pre %d", d);
    ds41_hc_post(y, res, post0, I, hc, dim, back);
    for (int i = 0; i < 64; i++) CHECK(back[i] == res[i], "hc_post identity %d", i);
}

static void test_rope(void) {
    printf("[rope] YaRN ramp: low dims untouched, high dims divided by factor\n");
    float f0[32], f1[32];
    ds41_rope_inv_freq(64, 160000.0f, 16.0f, 32, 1, 0, f0);
    ds41_rope_inv_freq(64, 160000.0f, 16.0f, 32, 1, 65536, f1);
    CHECK(f0[0] == 1.0f, "inv_freq[0] = 1");
    CHECK(fabsf(f1[0] - f0[0]) < 1e-9f, "fast dims keep frequency");
    CHECK(fabsf(f1[31] - f0[31] / 16.0f) < 1e-12f, "slow dims scaled by 1/factor: %g vs %g", f1[31], f0[31] / 16.0f);
    int mono = 1;
    for (int i = 1; i < 32; i++) if (f1[i] > f1[i - 1]) mono = 0;
    CHECK(mono, "inv_freq monotone decreasing");
}

/* Constants exported by export_ds41_engram_meta.py from config.json (tokenizer-independent). */
static const int64_t MULT[2][4] = {
    {76632096046245LL, 4839876093313LL, 35959672319349LL, 73987337458391LL},
    {67716810739261LL, 51510806800915LL, 30921347202721LL, 82619226485591LL}};
static const int64_t PRIMES[2][3][8] = {
    {{16000057, 16000079, 16000081, 16000097, 16000121, 16000129, 16000133, 16000183},
     {16000189, 16000207, 16000211, 16000253, 16000277, 16000289, 16000307, 16000321},
     {16000339, 16000381, 16000393, 16000399, 16000403, 16000409, 16000447, 16000463}},
    {{16000477, 16000487, 16000499, 16000507, 16000511, 16000573, 16000609, 16000627},
     {16000667, 16000669, 16000693, 16000697, 16000711, 16000729, 16000759, 16000769},
     {16000781, 16000799, 16000813, 16000819, 16000841, 16000877, 16000879, 16000889}}};
/* Expected rows (index 0, 7, 8, 23) computed by the Python reference for compressed ids
   [0, 1, 17, 99091, 2, 5000] with pad 2, positions 0..5. */
static const int64_t EXPECT[2][6][4] = {
    {{5702652, 121476532, 131476717, 380066193}, {15045429, 115577409, 128992384, 382841219},
     {11847637, 119508961, 141401859, 371455991}, {379248, 114357284, 129034992, 374200121},
     {13556909, 124480484, 135783393, 379397398}, {485543, 121962433, 136948829, 376066669}},
    {{14361964, 120604547, 132225184, 372375971}, {7992263, 125194906, 128857751, 373136088},
     {12090342, 126773197, 133923139, 382944738}, {4638160, 113837035, 131570239, 381974662},
     {2420072, 120815612, 138194782, 372498171}, {2515318, 114785177, 138144066, 376264123}}};

static void test_engram_hash(void) {
    printf("[engram] n-gram hash vs Python reference (embedded constants)\n");
    ds41_engram_meta m;
    memset(&m, 0, sizeof m);
    m.n_layers = 2; m.max_ngram = 4; m.n_heads = 8; m.pad_compressed_id = 2; m.compressed_vocab_size = 99092;
    memcpy(m.multipliers, MULT, sizeof MULT);
    memcpy(m.primes, PRIMES, sizeof PRIMES);
    for (int l = 0; l < 2; l++) {
        int64_t acc = 0;
        for (int i = 0; i < 24; i++) { m.offsets[l][i] = acc; acc += PRIMES[l][i / 8][i % 8]; }
        CHECK(acc == (l == 0 ? 384006168LL : 384016682LL), "table rows layer %d = %lld", l, (long long)acc);
    }
    const int32_t comp[6] = {0, 1, 17, 99091, 2, 5000};
    for (int l = 0; l < 2; l++)
        for (int pos = 0; pos < 6; pos++) {
            int32_t hist[4];
            for (int s = 0; s < 4; s++) hist[s] = pos - s >= 0 ? comp[pos - s] : -1;
            int64_t rows[24];
            ds41_engram_hash_compressed(&m, l, hist, rows);
            CHECK(rows[0] == EXPECT[l][pos][0] && rows[7] == EXPECT[l][pos][1] &&
                  rows[8] == EXPECT[l][pos][2] && rows[23] == EXPECT[l][pos][3],
                  "engram L%d pos%d: %lld %lld %lld %lld", l, pos,
                  (long long)rows[0], (long long)rows[7], (long long)rows[8], (long long)rows[23]);
            for (int i = 0; i < 24; i++)
                CHECK(rows[i] >= m.offsets[l][i] && rows[i] < m.offsets[l][i] + PRIMES[l][i / 8][i % 8],
                      "row %d out of its bucket", i);
        }

    printf("[engram] row dequant + gate\n");
    uint8_t row[256], sc[8];
    for (int i = 0; i < 256; i++) row[i] = 0x38;        /* 1.0 */
    for (int b = 0; b < 8; b++) sc[b] = (uint8_t)(127 + b);
    float out[256];
    ds41_engram_row_dequant(row, sc, out);
    for (int i = 0; i < 256; i++) CHECK(out[i] == ldexpf(1.0f, i / 32), "row dequant %d", i);
    float h[8] = {1, 1, 1, 1, 1, 1, 1, 1}, key[8] = {1, 1, 1, 1, 1, 1, 1, 1}, val[8], qk[8];
    for (int i = 0; i < 8; i++) { val[i] = 2.0f; qk[i] = 1.0f; }
    ds41_engram_apply(h, key, val, qk, 8, DS41_NORM_EPS);
    /* dot = 8 * rstd(=1) / sqrt(8) = sqrt(8); gate = sigmoid(sqrt(sqrt(8))) */
    float gate = 1.0f / (1.0f + expf(-sqrtf(sqrtf(8.0f))));
    for (int i = 0; i < 8; i++) CHECK(fabsf(h[i] - (1.0f + gate * 2.0f)) < 1e-6f, "engram apply %d: %g", i, h[i]);
}

static int engram_file_mode(int argc, char **argv) {
    ds41_engram_meta m;
    int rc = ds41_engram_meta_load(argv[2], &m);
    if (rc) { fprintf(stderr, "cannot load %s (rc=%d)\n", argv[2], rc); return 1; }
    printf("[engram] loaded %s: vocab %u -> %u compressed, pad %u, layers %d,%d\n", argv[2],
           m.vocab_size, m.compressed_vocab_size, m.pad_compressed_id, m.layer_ids[0], m.layer_ids[1]);
    int n = argc - 3;
    int32_t *ids = malloc(sizeof(int32_t) * n);
    for (int i = 0; i < n; i++) ids[i] = atoi(argv[3 + i]);
    for (uint32_t l = 0; l < m.n_layers; l++)
        for (int pos = 0; pos < n; pos++) {
            int32_t hist[4];
            for (int s = 0; s < 4; s++) hist[s] = pos - s >= 0 ? ids[pos - s] : -1;
            int64_t rows[24];
            ds41_engram_hash(&m, (int)l, hist, rows);
            printf("L%u pos%d", l, pos);
            for (int i = 0; i < 24; i++) printf(" %lld", (long long)rows[i]);
            printf("\n");
        }
    free(ids);
    ds41_engram_meta_free(&m);
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 4 && strcmp(argv[1], "--engram") == 0) return engram_file_mode(argc, argv);
    test_scalar_decoders();
    test_mxfp4_matvec();
    test_fp8blk_matvec();
    test_swiglu_and_norm();
    test_routing();
    test_hc();
    test_rope();
    test_engram_hash();
    if (g_fail) { printf("\n%d FAILURE(S)\n", g_fail); return 1; }
    printf("\nALL ds41 TESTS PASSED\n");
    return 0;
}
