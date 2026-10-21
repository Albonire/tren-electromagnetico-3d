# Tren Electromagnetico 3D - Simulacion Fisica

Simulacion fisica 3D interactiva del tren electromagnetico homopolar con pila, imanes de neodimio, bobina helicoidal conductora y campos magneticos explicitos. Construido con Three.js, React y Vite.

## Ejecucion Local

1. Instalar dependencias:
   ```bash
   bun install # o npm install / pnpm install
   ```

2. Ejecutar servidor de desarrollo:
   ```bash
   bun run dev # o npm run dev
   ```

3. Abrir en el navegador:
   `http://localhost:3000` (o `http://localhost:3001` si el puerto 3000 esta en uso)

## Modelo físico

La pila y los dos grupos de imanes cierran un circuito a través de las espiras de cobre que quedan entre los imanes. Ese tramo de bobina forma un solenoide corto con `B = μ0 · n · I`, y cada imán, situado en un extremo, recibe un empuje `F = k · I`. La misma constante `k` produce la fuerza contraelectromotriz `k · v`, que reduce la corriente a medida que el tren acelera.

- Corriente: `I = (1.5 V − k · v) / R`, con `R ≈ 0.65 Ω` (resistencia interna de la pila más los contactos).
- Pérdidas: rozamiento de Coulomb y amortiguamiento proporcional a la velocidad por el contacto con las espiras.
- Sobre la regleta del hueco el circuito se abre: no hay corriente ni fuerza, y el tren cruza solo por inercia.
- Invertir la polaridad conserva el momento: el tren frena y luego acelera en sentido contrario.
- La velocidad real ronda los 0.7 m/s; la escala de tiempo por defecto (0.1×) permite seguirla.

Las constantes están en `PHYSICS` dentro de `src/simulation/physics.ts` y los tests en `src/simulation/physics.test.ts` (`npx tsx --test src/simulation/physics.test.ts`).
