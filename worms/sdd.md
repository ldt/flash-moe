# Software Design Document: WormsJS

## 1. Overview

### 1.1 Purpose
WormsJS is a browser-based, turn-based artillery tactical game inspired by the classic
**Worms** series by Team17. Players control a team of worms and take turns firing
projectiles at an opposing team across destructible terrain, using physics-based
trajectory calculation and a variety of weapons.

### 1.2 Scope
- Two teams of worms (Red vs Blue), each with configurable worm count.
- Turn-based combat with a timer per turn.
- Physics-based projectile motion with gravity, wind, and collision detection.
- Destructible pixel-based terrain.
- Multiple weapons: Bazooka, Grenade, and Shotgun.
- Explosive damage with falloff based on distance from blast epicenter.
- Worm death and win conditions (eliminate all enemy worms).
- Simple AI for the enemy team.
- Fully self-contained single HTML file (no external dependencies).

### 1.3 Target Platform
- Modern web browsers supporting HTML5 Canvas and ES6+ JavaScript.
- No server-side components required.

---

## 2. Game Mechanics

### 2.1 Turn System
- Players alternate turns. On a player's turn, they may:
  1. Aim the weapon (mouse drag to set angle and power).
  2. Fire the weapon.
  3. Wait for all projectiles to settle.
- After the player's turn ends, the AI team takes its turn automatically.
- A turn timer (default 30 seconds) counts down; if it reaches zero, the turn ends
  automatically (the worm skips its shot).

### 2.2 Projectile Physics
- **Gravity**: Constant downward acceleration applied each frame (`g = 0.2 px/frame²`).
- **Wind**: A horizontal force that varies per turn, applied as a small acceleration
  to the projectile's x-velocity.
- **Velocity**: Each projectile has an initial velocity determined by aim angle and power.
- **Integration**: Semi-implicit Euler integration for position and velocity updates.

### 2.3 Collision Detection
- **Terrain Collision**: Checked via pixel sampling on the terrain canvas. When a
  projectile's position lands on a non-transparent pixel, it explodes.
- **Worm Collision**: Checked via circle-to-circle collision. If a projectile's
  velocity is low enough and it touches a worm, it explodes.
- **Boundary**: Projectiles that fly off-screen are removed.

### 2.4 Damage System
- **Explosion**: When a projectile explodes, all worms within a blast radius take
  damage. Damage falls off linearly with distance from the epicenter.
  - Formula: `damage = maxDamage * (1 - distance / radius)` clamped to `[0, maxDamage]`.
- **Instant Death**: Worms reduced to 0 HP are removed from the game.
- **Fall Damage**: Worms that fall from a great height take damage based on fall
  velocity.

### 2.5 Terrain Destruction
- When an explosion occurs, terrain pixels within the blast radius are set to
  transparent using canvas `globalCompositeOperation = 'destination-out'`.
- A circular "hole" is carved into the terrain.
- Worms standing on destroyed terrain may fall if their support is removed.

### 2.6 Win Condition
- The game ends when all worms on one team are eliminated.
- The remaining team wins.

---

## 3. Architecture

### 3.1 File Structure
```
worms.html          # Single self-contained HTML file
  ├── <style>       # CSS for canvas and UI
  └── <script>      # All JavaScript game logic
```

### 3.2 Module Layout (within the single JS block)

| Component         | Responsibility                                         |
|-------------------|--------------------------------------------------------|
| Constants         | Game-wide configuration values                         |
| GameState         | Central state: current turn, phase, winner               |
| Terrain           | Terrain generation, rendering, pixel-level destruction |
| Worm              | Worm entity: position, health, team, AI logic          |
| Projectile        | Projectile entity: position, velocity, weapon type     |
| WeaponSystem      | Weapon definitions, firing logic, explosion handling   |
| Physics           | Gravity, wind, collision detection utilities           |
| Input             | Mouse/touch handling for aiming and firing             |
| Renderer          | Canvas drawing: terrain, worms, projectiles, UI        |
| GameLoop          | Main loop using requestAnimationFrame                  |

### 3.3 Data Flow

```
User Input (mouse) → Input → GameState (set aim/power) → WeaponSystem.fire()
  → Projectile created → GameLoop updates Projectile (Physics)
  → Collision check → Explosion (Terrain destruction + Damage)
  → GameState checks win condition → Next turn
  → Renderer draws everything each frame
```

---

## 4. Detailed Component Design

### 4.1 Constants
```javascript
const CONFIG = {
  canvasWidth: 1024,
  canvasHeight: 576,
  gravity: 0.2,
  windMin: -1.5,
  windMax: 1.5,
  turnTime: 30,          // seconds
  wormHealth: 100,
  wormRadius: 10,
  explosionRadius: 30,
  explosionMaxDamage: 50,
  terrainThickness: 40,
  numWormsPerTeam: 5,
  weapons: {
    bazooka: { speed: 8, damage: 50, radius: 30, gravity: true },
    grenade: { speed: 6, damage: 40, radius: 35, gravity: true, fuse: 3 },
    shotgun: { speed: 12, damage: 20, radius: 15, gravity: false, pellets: 5 }
  }
};
```

### 4.2 GameState
Manages:
- `currentTeam`: which team's turn it is.
- `currentWormIndex`: index of the active worm on the current team.
- `phase`: 'AIMING' | 'FIRING' | 'EXPLODING' | 'WAITING' | 'GAME_OVER'.
- `turnTimer`: countdown timer.
- `projectiles`: array of active projectiles.
- `winner`: null or team color.
- Methods: `nextTurn()`, `endTurn()`, `checkWinCondition()`.

### 4.3 Terrain
- **Generation**: Uses layered sine waves to create a natural-looking landscape.
  The terrain is drawn to an offscreen canvas as ImageData.
- **Rendering**: Blitted to the main canvas each frame.
- **Destruction**: Uses `ctx.globalCompositeOperation = 'destination-out'` with a
  circular brush to carve holes.
- **Pixel Sampling**: `getPixel(x, y)` returns whether a pixel is solid (non-transparent).

### 4.4 Worm
```javascript
class Worm {
  constructor(x, y, team) {
    this.x, this.y, this.team, this.health, this.radius
    this.angle, this.power, this.isAI
  }
  update()       // physics (falling, etc.)
  draw(ctx)      // render
  takeDamage(d)  // reduce health, check death
  aiAim()        // AI: calculate aim toward target
  aiFire()       // AI: execute fire
}
```

### 4.5 Projectile
```javascript
class Projectile {
  constructor(x, y, vx, vy, weaponType, owner) {
    this.x, this.y, this.vx, this.vy
    this.weaponType, this.owner
    this.life, this.fuse  // for grenades
  }
  update()       // apply physics, check collisions
  draw(ctx)      // render
  explode()      // trigger explosion
}
```

### 4.6 WeaponSystem
- `fire(worm, angle, power)`: Creates a projectile with the appropriate velocity.
- `explosion(x, y, weaponType)`: Handles terrain destruction and damage application.
- `applyDamage(x, y, radius, maxDamage)`: Iterates over all worms, applies falloff damage.

### 4.7 Physics
- `applyGravity(projectile)`: Adds gravity to vy.
- `applyWind(projectile, wind)`: Adds wind to vx.
- `checkTerrainCollision(projectile)`: Samples terrain pixels.
- `checkWormCollision(projectile)`: Circle-to-circle check.

### 4.8 Input
- **MouseDown**: Start aiming (record start position).
- **MouseMove**: Update aim angle and power (visualized as a line).
- **MouseUp**: Fire the weapon with calculated angle and power.
- **Keyboard**: Spacebar to skip turn (debug/dev convenience).

### 4.9 Renderer
- Draws terrain (blit from offscreen canvas).
- Draws worms (circles with team color, health bar).
- Draws projectiles (small circles).
- Draws UI: aim line, power bar, turn indicator, health bars, timer.
- Draws explosion effects (expanding circle, particles).

### 4.10 GameLoop
```javascript
function gameLoop() {
  update();    // update physics, AI, timers
  render();    // draw everything
  requestAnimationFrame(gameLoop);
}
```

---

## 5. Rendering Pipeline

1. Clear main canvas.
2. Draw background (sky gradient).
3. Blit terrain canvas.
4. Draw active projectiles.
5. Draw worms (with health bars).
6. Draw explosion effects.
7. Draw UI overlay (aim line, power, timer, turn info).

---

## 6. Edge Cases & Error Handling

- **Projectile stuck in terrain**: Explode on contact.
- **Worm falls off map**: Remove from game (counts as death).
- **All worms on a team dead simultaneously**: Game over.
- **Turn timer expires during explosion**: Wait for explosion to finish before
  switching turns.
- **Division by zero in aim calculation**: Guard against zero-length aim vectors.

---

## 7. Performance Considerations

- Terrain is rendered once to an offscreen canvas and only modified (not regenerated)
  during destruction.
- Projectile count is limited (one per worm per turn), so collision checks are cheap.
- Worm count is small (5 per team), so damage iteration is trivial.
- requestAnimationFrame ensures smooth 60fps rendering.

---

## 8. Future Enhancements (Not in Scope)

- Additional weapons (banana bomb, holy hand grenade, etc.).
- Rope and ninja rope mechanics.
- Multiplayer over WebSocket.
- Sound effects and music.
- Particle systems for explosions and smoke trails.
- Map selection and custom maps.
