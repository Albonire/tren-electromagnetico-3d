import { useEffect, useRef, useState, type ReactNode } from 'react';
import { AnimatePresence, MotionConfig, motion } from 'motion/react';
import {
  ArrowDownToDot,
  ArrowLeftRight,
  Check,
  CircleQuestionMark,
  Magnet,
  Navigation,
  Orbit,
  Pause,
  PictureInPicture2,
  Play,
  ScanEye,
  Video,
  X,
  type LucideIcon,
} from 'lucide-react';
import { PIP_BEZEL, type PipLayout } from './layout';

export type CameraMode = 'overview' | 'pov' | 'follow' | 'top' | 'tunnel';

export const CAMERA_MODES: ReadonlyArray<{
  mode: CameraMode;
  label: string;
  shortcut: string;
  Icon: LucideIcon;
}> = [
  { mode: 'overview', label: 'General', shortcut: '1', Icon: Orbit },
  { mode: 'follow', label: 'Seguir', shortcut: '2', Icon: Navigation },
  { mode: 'top', label: 'Cenital', shortcut: '3', Icon: ArrowDownToDot },
  { mode: 'tunnel', label: 'Frontal', shortcut: '4', Icon: Video },
  { mode: 'pov', label: 'A bordo', shortcut: '5', Icon: ScanEye },
];

/** Simulation time scale (1 = real time). The real train covers the track in about 1.5 s. */
export const SPEED_PRESETS: ReadonlyArray<{ label: string; value: number }> = [
  { label: '0.1×', value: 0.1 },
  { label: '0.25×', value: 0.25 },
  { label: '0.5×', value: 0.5 },
  { label: '1×', value: 1 },
];

export function nearestSpeedPresetIndex(multiplier: number): number {
  let best = 0;
  SPEED_PRESETS.forEach((p, i) => {
    if (Math.abs(p.value - multiplier) < Math.abs(SPEED_PRESETS[best].value - multiplier)) {
      best = i;
    }
  });
  return best;
}

export interface SimulationUiState {
  isPaused: boolean;
  cameraMode: CameraMode;
  showExplicitFields: boolean;
  showPip: boolean;
  direction: 1 | -1;
  speedMultiplier: number;
  currentAmps: number;
  magneticFieldmT: number;
  propulsionForceN: number;
  speedCmPerS: number;
  lapCount: number;
}

export interface SimulationUiActions {
  togglePause: () => void;
  reversePolarity: () => void;
  toggleFields: () => void;
  togglePip: () => void;
  setCameraMode: (mode: CameraMode) => void;
  setSpeed: (multiplier: number) => void;
}

export interface CalloutInfo {
  id: string;
  symbol: string;
  title: string;
  subtitle: string;
  accent: string;
}

// Apple-style dark material: translucent fill, heavy blur, hairline inner stroke, soft drop shadow.
const GLASS =
  'bg-[rgba(30,30,32,0.66)] backdrop-blur-2xl backdrop-saturate-[1.8] shadow-[0_12px_32px_-10px_rgba(0,0,0,0.6),inset_0_0_0_0.5px_rgba(255,255,255,0.16)]';
const FOCUS_RING =
  'focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#0a84ff] focus-visible:ring-offset-0';
const SPRING = { type: 'spring', stiffness: 520, damping: 40, mass: 0.9 } as const;

function Kbd({ children }: { children: ReactNode }) {
  return (
    <kbd className="inline-grid min-w-[22px] h-[22px] place-items-center rounded-[6px] bg-white/[0.1] px-1.5 font-sans text-[12px] font-medium text-white/80 shadow-[inset_0_-1px_0_rgba(255,255,255,0.06)]">
      {children}
    </kbd>
  );
}

function Tooltip({ label, shortcut }: { label: string; shortcut?: string }) {
  return (
    <span
      role="tooltip"
      className="pointer-events-none absolute bottom-full left-1/2 z-30 mb-3 hidden -translate-x-1/2 translate-y-1 items-center gap-2 whitespace-nowrap rounded-[10px] bg-[rgba(44,44,46,0.96)] px-2.5 py-1.5 text-[12px] font-medium text-white/90 opacity-0 shadow-[0_8px_24px_-6px_rgba(0,0,0,0.6),inset_0_0_0_0.5px_rgba(255,255,255,0.14)] transition-all duration-150 md:flex group-hover:translate-y-0 group-hover:opacity-100 group-hover:delay-500"
    >
      {label}
      {shortcut && <Kbd>{shortcut}</Kbd>}
    </span>
  );
}

function IconButton({
  label,
  shortcut,
  pressed,
  prominent,
  onClick,
  children,
}: {
  label: string;
  shortcut?: string;
  pressed?: boolean;
  prominent?: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  const tone = prominent
    ? 'bg-white text-black hover:bg-white/90'
    : pressed
      ? 'bg-white/[0.18] text-white'
      : 'text-white/70 hover:bg-white/[0.08] hover:text-white';
  return (
    <span className="group relative">
      <button
        type="button"
        aria-label={label}
        aria-pressed={pressed}
        onClick={onClick}
        className={`grid size-10 cursor-pointer place-items-center rounded-full transition-colors duration-200 active:scale-[0.96] ${tone} ${FOCUS_RING}`}
      >
        {children}
      </button>
      <Tooltip label={label} shortcut={shortcut} />
    </span>
  );
}

function Separator() {
  return <span aria-hidden="true" className="mx-1 h-6 w-px shrink-0 bg-white/[0.12]" />;
}

function Segmented<T extends string | number>({
  id,
  ariaLabel,
  value,
  options,
  onChange,
  showLabels = true,
}: {
  id: string;
  ariaLabel: string;
  value: T;
  options: ReadonlyArray<{ value: T; label: string; shortcut?: string; Icon?: LucideIcon }>;
  onChange: (value: T) => void;
  showLabels?: boolean;
}) {
  return (
    <div role="radiogroup" aria-label={ariaLabel} className="flex rounded-full bg-white/[0.07] p-[3px]">
      {options.map(({ value: v, label, shortcut, Icon }) => {
        const selected = v === value;
        return (
          <span key={String(v)} className="group relative">
            <button
              type="button"
              role="radio"
              aria-checked={selected}
              aria-label={label}
              onClick={() => onChange(v)}
              className={`relative flex h-[34px] cursor-pointer items-center rounded-full text-[13px] font-medium transition-colors duration-200 ${
                showLabels || !Icon ? 'px-3' : 'w-[38px] justify-center'
              } ${selected ? 'text-white' : 'text-white/60 hover:text-white/90'} ${FOCUS_RING}`}
            >
              {selected && (
                <motion.span
                  layoutId={`${id}-thumb`}
                  transition={SPRING}
                  className="absolute inset-0 rounded-full bg-white/[0.2] shadow-[0_1px_3px_rgba(0,0,0,0.35),inset_0_0_0_0.5px_rgba(255,255,255,0.2)]"
                />
              )}
              <span className="relative flex items-center gap-1.5 tabular-nums">
                {Icon && <Icon className="size-[15px]" strokeWidth={1.9} />}
                {(showLabels || !Icon) && label}
              </span>
            </button>
            {Icon && !showLabels && <Tooltip label={label} shortcut={shortcut} />}
          </span>
        );
      })}
    </div>
  );
}

function CameraMenu({
  value,
  onChange,
}: {
  value: CameraMode;
  onChange: (mode: CameraMode) => void;
}) {
  const [open, setOpen] = useState(false);
  const current = CAMERA_MODES.find((m) => m.mode === value) ?? CAMERA_MODES[0];
  return (
    <span className="relative">
      <button
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Cámara: ${current.label}`}
        onClick={() => setOpen((o) => !o)}
        className={`grid size-10 cursor-pointer place-items-center rounded-full text-white/85 hover:bg-white/[0.08] ${open ? 'bg-white/[0.12]' : ''} ${FOCUS_RING}`}
      >
        <current.Icon className="size-[18px]" strokeWidth={1.9} />
      </button>
      <AnimatePresence>
        {open && (
          <>
            <span className="fixed inset-0 z-20" onClick={() => setOpen(false)} aria-hidden="true" />
            <motion.div
              role="menu"
              initial={{ opacity: 0, y: 8, scale: 0.96 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 6, scale: 0.97 }}
              transition={{ duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}
              className={`absolute bottom-full left-1/2 z-30 mb-3 w-48 origin-bottom -translate-x-1/2 overflow-hidden rounded-[14px] p-1 ${GLASS}`}
            >
              {CAMERA_MODES.map(({ mode, label, Icon }) => (
                <button
                  key={mode}
                  type="button"
                  role="menuitemradio"
                  aria-checked={mode === value}
                  onClick={() => {
                    onChange(mode);
                    setOpen(false);
                  }}
                  className={`flex w-full cursor-pointer items-center gap-2.5 rounded-[10px] px-2.5 py-2 text-left text-[14px] text-white/90 hover:bg-white/[0.08] ${FOCUS_RING}`}
                >
                  <Icon className="size-[17px] text-white/60" strokeWidth={1.9} />
                  <span className="flex-1">{label}</span>
                  {mode === value && <Check className="size-4 text-[#0a84ff]" strokeWidth={2.4} />}
                </button>
              ))}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </span>
  );
}

function Dock({
  ui,
  actions,
  compact,
  wide,
  onOpenHelp,
}: {
  ui: SimulationUiState;
  actions: SimulationUiActions;
  compact: boolean;
  wide: boolean;
  onOpenHelp: () => void;
}) {
  const speedIndex = nearestSpeedPresetIndex(ui.speedMultiplier);
  return (
    <nav
      aria-label="Controles de la simulación"
      className={`pointer-events-auto flex h-14 items-center gap-0.5 rounded-full px-2 ${GLASS}`}
    >
      <IconButton
        label={ui.isPaused ? 'Reanudar' : 'Pausar'}
        shortcut="Espacio"
        prominent
        onClick={actions.togglePause}
      >
        {ui.isPaused ? (
          <Play className="size-[18px] translate-x-[1px] fill-current" strokeWidth={0} />
        ) : (
          <Pause className="size-[18px] fill-current" strokeWidth={0} />
        )}
      </IconButton>
      <IconButton label="Invertir polaridad" shortcut="R" onClick={actions.reversePolarity}>
        <ArrowLeftRight className="size-[18px]" strokeWidth={1.9} />
      </IconButton>
      <IconButton
        label={ui.showExplicitFields ? 'Ocultar campos' : 'Mostrar campos'}
        shortcut="F"
        pressed={ui.showExplicitFields}
        onClick={actions.toggleFields}
      >
        <Magnet className="size-[18px]" strokeWidth={1.9} />
      </IconButton>

      <Separator />

      {compact ? (
        <CameraMenu value={ui.cameraMode} onChange={actions.setCameraMode} />
      ) : (
        <Segmented
          id="camera"
          ariaLabel="Cámara"
          value={ui.cameraMode}
          onChange={actions.setCameraMode}
          showLabels={wide}
          options={CAMERA_MODES.map(({ mode, label, shortcut, Icon }) => ({
            value: mode,
            label,
            shortcut,
            Icon,
          }))}
        />
      )}

      <Separator />

      {compact ? (
        <button
          type="button"
          aria-label={`Escala de tiempo ${SPEED_PRESETS[speedIndex].label}. Toca para cambiar`}
          onClick={() =>
            actions.setSpeed(SPEED_PRESETS[(speedIndex + 1) % SPEED_PRESETS.length].value)
          }
          className={`h-10 min-w-11 cursor-pointer rounded-full px-2.5 text-[13px] font-semibold tabular-nums text-white/85 hover:bg-white/[0.08] ${FOCUS_RING}`}
        >
          {SPEED_PRESETS[speedIndex].label}
        </button>
      ) : (
        <Segmented
          id="speed"
          ariaLabel="Escala de tiempo"
          value={speedIndex}
          onChange={(i) => actions.setSpeed(SPEED_PRESETS[i].value)}
          options={SPEED_PRESETS.map((p, i) => ({ value: i, label: p.label }))}
        />
      )}

      <Separator />

      <IconButton
        label={ui.showPip ? 'Ocultar vista secundaria' : 'Mostrar vista secundaria'}
        shortcut="V"
        pressed={ui.showPip}
        onClick={actions.togglePip}
      >
        <PictureInPicture2 className="size-[18px]" strokeWidth={1.9} />
      </IconButton>
      <IconButton label="Ayuda y atajos" shortcut="?" onClick={onOpenHelp}>
        <CircleQuestionMark className="size-[18px]" strokeWidth={1.9} />
      </IconButton>
    </nav>
  );
}

function Metric({
  label,
  value,
  unit,
  dot,
  compact,
}: {
  label: string;
  value: string;
  unit: string;
  dot: string;
  compact: boolean;
}) {
  return (
    <div className="min-w-0">
      <div className="flex items-center gap-1.5 text-[11px] font-medium text-white/50">
        <span className="size-1.5 shrink-0 rounded-full" style={{ backgroundColor: dot }} />
        <span className="truncate">{label}</span>
      </div>
      <div
        className={`mt-0.5 font-semibold tabular-nums tracking-[-0.02em] text-white ${
          compact ? 'text-[16px]' : 'text-[22px] leading-7'
        }`}
      >
        {value}
        <span className={`ml-1 font-normal tracking-normal text-white/45 ${compact ? 'text-[11px]' : 'text-[13px]'}`}>
          {unit}
        </span>
      </div>
    </div>
  );
}

function TelemetryCard({ ui, compact }: { ui: SimulationUiState; compact: boolean }) {
  const circuitClosed = ui.currentAmps > 0.05;
  const metrics = (
    <>
      <Metric compact={compact} label="Corriente" value={ui.currentAmps.toFixed(2)} unit="A" dot="#64d2ff" />
      <Metric compact={compact} label="Campo" value={ui.magneticFieldmT.toFixed(1)} unit="mT" dot="#0a84ff" />
      <Metric compact={compact} label="Fuerza" value={ui.propulsionForceN.toFixed(2)} unit="N" dot="#30d158" />
      <Metric compact={compact} label="Velocidad" value={ui.speedCmPerS.toFixed(0)} unit="cm/s" dot="#ffd60a" />
    </>
  );

  if (compact) {
    return (
      <section
        aria-label="Telemetría"
        className={`pointer-events-auto grid grid-cols-4 gap-3 rounded-[18px] px-3.5 py-2.5 ${GLASS}`}
      >
        {metrics}
      </section>
    );
  }

  return (
    <section aria-label="Telemetría" className={`pointer-events-auto w-[300px] rounded-[22px] p-4 ${GLASS}`}>
      <header className="flex items-start justify-between gap-3">
        <div>
          <h1 className="text-[15px] font-semibold tracking-[-0.01em] text-white">Tren electromagnético</h1>
          <p className="mt-0.5 text-[12px] text-white/50">Pila AA, imanes y bobina de cobre</p>
        </div>
      </header>
      <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3.5">{metrics}</div>
      <footer className="mt-4 flex items-center justify-between border-t border-white/[0.08] pt-3 text-[12px] text-white/55">
        <span className="flex items-center gap-1.5">
          <span
            className={`size-1.5 rounded-full transition-colors duration-300 ${
              circuitClosed ? 'bg-[#30d158] shadow-[0_0_8px_#30d158]' : 'bg-white/30'
            }`}
          />
          {circuitClosed ? 'Circuito cerrado' : 'Circuito abierto'}
        </span>
        <span className="tabular-nums">
          {ui.lapCount} {ui.lapCount === 1 ? 'vuelta' : 'vueltas'} · {ui.direction === 1 ? 'Polaridad +' : 'Polaridad −'}
        </span>
      </footer>
    </section>
  );
}

function PipFrame({
  layout,
  cameraMode,
  onSwap,
  onClose,
}: {
  layout: PipLayout;
  cameraMode: CameraMode;
  onSwap: () => void;
  onClose: () => void;
}) {
  const showsOverview = cameraMode === 'pov';
  return (
    <div
      className="group pointer-events-auto absolute rounded-[18px] shadow-[0_18px_40px_-12px_rgba(0,0,0,0.7)]"
      style={{
        left: layout.left - PIP_BEZEL,
        top: layout.top - PIP_BEZEL,
        width: layout.width + PIP_BEZEL * 2,
        height: layout.height + PIP_BEZEL * 2,
        border: `${PIP_BEZEL}px solid rgba(28,28,30,0.96)`,
        outline: '0.5px solid rgba(255,255,255,0.16)',
      }}
    >
      <button
        type="button"
        onClick={onSwap}
        aria-label={showsOverview ? 'Volver a la vista general' : 'Ver a bordo en pantalla completa'}
        className={`absolute inset-0 cursor-pointer rounded-[14px] ${FOCUS_RING}`}
      />
      <span className="pointer-events-none absolute left-2 top-2 rounded-full bg-black/45 px-2 py-0.5 text-[11px] font-medium text-white/90 backdrop-blur-md">
        {showsOverview ? 'General' : 'A bordo'}
      </span>
      <button
        type="button"
        onClick={onClose}
        aria-label="Ocultar vista secundaria"
        className={`absolute right-2 top-2 grid size-6 cursor-pointer place-items-center rounded-full bg-black/45 text-white/80 opacity-0 backdrop-blur-md transition-opacity duration-200 hover:text-white group-hover:opacity-100 focus-visible:opacity-100 ${FOCUS_RING}`}
      >
        <X className="size-3.5" strokeWidth={2.2} />
      </button>
    </div>
  );
}

function HelpSheet({ open, onClose }: { open: boolean; onClose: () => void }) {
  const closeRef = useRef<HTMLButtonElement | null>(null);
  useEffect(() => {
    if (open) closeRef.current?.focus();
  }, [open]);

  const gestures: Array<[string, ReactNode]> = [
    ['Mover o lanzar el tren', 'Arrastra la pila y suéltala'],
    ['Colocar el tren', <><Kbd>Shift</Kbd> + clic en la bobina</>],
    ['Invertir polaridad', 'Doble clic en el tren'],
    ['Orbitar y acercar', 'Arrastra el fondo, rueda o pellizco'],
  ];
  const keys: Array<[string, ReactNode]> = [
    ['Pausar o reanudar', <Kbd>Espacio</Kbd>],
    ['Invertir polaridad', <Kbd>R</Kbd>],
    ['Campos y electrones', <Kbd>F</Kbd>],
    ['Cámaras', <span className="flex gap-1"><Kbd>1</Kbd><Kbd>2</Kbd><Kbd>3</Kbd><Kbd>4</Kbd><Kbd>5</Kbd></span>],
    ['Escala de tiempo', <span className="flex gap-1"><Kbd>↑</Kbd><Kbd>↓</Kbd></span>],
    ['Vista secundaria', <Kbd>V</Kbd>],
  ];

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="pointer-events-auto fixed inset-0 z-40 grid place-items-center bg-black/35 p-4 backdrop-blur-[2px]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          onClick={onClose}
        >
          <motion.div
            role="dialog"
            aria-modal="true"
            aria-labelledby="help-title"
            initial={{ opacity: 0, y: 16, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.98 }}
            transition={{ duration: 0.28, ease: [0.2, 0.8, 0.2, 1] }}
            onClick={(e) => e.stopPropagation()}
            className={`max-h-[calc(100dvh-32px)] w-full max-w-[420px] overflow-y-auto rounded-[24px] p-5 ${GLASS}`}
          >
            <header className="flex items-start justify-between gap-4">
              <div>
                <h2 id="help-title" className="text-[17px] font-semibold tracking-[-0.01em] text-white">
                  Cómo funciona
                </h2>
                <p className="mt-1.5 text-[13px] leading-[1.45] text-white/60">
                  Los imanes conducen la corriente de la pila a través de las espiras que tocan. Ese tramo de
                  bobina se convierte en un electroimán que empuja a los imanes, y el tren avanza solo. Sobre
                  la regleta el circuito se abre: el tren cruza el hueco solo por inercia.
                </p>
              </div>
              <button
                ref={closeRef}
                type="button"
                onClick={onClose}
                aria-label="Cerrar"
                className={`grid size-7 shrink-0 cursor-pointer place-items-center rounded-full bg-white/[0.1] text-white/70 hover:bg-white/[0.16] hover:text-white ${FOCUS_RING}`}
              >
                <X className="size-4" strokeWidth={2.2} />
              </button>
            </header>

            {[
              ['Gestos', gestures],
              ['Teclado', keys],
            ].map(([title, rows]) => (
              <section key={title as string} className="mt-5">
                <h3 className="px-1 text-[12px] font-medium text-white/45">{title as string}</h3>
                <dl className="mt-1.5 divide-y divide-white/[0.07] rounded-[14px] bg-white/[0.05]">
                  {(rows as Array<[string, ReactNode]>).map(([label, value]) => (
                    <div key={label} className="flex items-center justify-between gap-4 px-3.5 py-2.5 text-[13px]">
                      <dt className="text-white/90">{label}</dt>
                      <dd className="flex items-center gap-1 text-right text-white/55">{value}</dd>
                    </div>
                  ))}
                </dl>
              </section>
            ))}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

function Hint({ show, text }: { show: boolean; text: string }) {
  return (
    <AnimatePresence>
      {show && (
        <motion.p
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 4 }}
          transition={{ duration: 0.3 }}
          className={`rounded-full px-3.5 py-1.5 text-[12px] text-white/75 ${GLASS}`}
        >
          {text}
        </motion.p>
      )}
    </AnimatePresence>
  );
}

export function CalloutLabels({
  callouts,
  registerRef,
  compact,
}: {
  callouts: CalloutInfo[];
  registerRef: (id: string, el: HTMLDivElement | null) => void;
  compact: boolean;
}) {
  return (
    <>
      {callouts.map((c) => (
        <div
          key={c.id}
          ref={(el) => registerRef(c.id, el)}
          aria-hidden="true"
          className="pointer-events-none absolute left-0 top-0 opacity-0 transition-opacity duration-300 will-change-transform"
        >
          <div
            className={`mb-1.5 flex items-center gap-2 whitespace-nowrap rounded-full p-1 ${compact ? '' : 'pr-3'} ${GLASS}`}
          >
            <span
              className="grid size-6 place-items-center rounded-full bg-white/[0.08] text-[12px] font-semibold"
              style={{ color: c.accent }}
            >
              {c.symbol}
            </span>
            {!compact && (
              <>
                <span className="text-[12px] font-medium text-white/90">{c.title}</span>
                <span className="text-[12px] text-white/45">{c.subtitle}</span>
              </>
            )}
          </div>
        </div>
      ))}
    </>
  );
}

export function SimulationUi({
  ui,
  actions,
  compact,
  wide,
  pipLayout,
  helpOpen,
  setHelpOpen,
}: {
  ui: SimulationUiState;
  actions: SimulationUiActions;
  compact: boolean;
  wide: boolean;
  pipLayout: PipLayout;
  helpOpen: boolean;
  setHelpOpen: (open: boolean) => void;
}) {
  const [showHint, setShowHint] = useState(true);
  // With the circuit open there is no force: a train that stops over the gap stays there
  const stalled = !ui.isPaused && ui.currentAmps <= 0.05 && ui.speedCmPerS < 0.5;
  const stallText = 'Circuito abierto: arrastra la pila y suéltala para lanzarla';
  useEffect(() => {
    const t = window.setTimeout(() => setShowHint(false), 7000);
    return () => window.clearTimeout(t);
  }, []);

  return (
    <MotionConfig reducedMotion="user">
      <div className="pointer-events-none absolute inset-0 z-10 font-sans text-white antialiased">
        <div className={`absolute flex flex-col items-center gap-2 ${compact ? 'inset-x-3 top-3' : 'left-5 top-5'}`}>
          <TelemetryCard ui={ui} compact={compact} />
          {compact && (
            <Hint
              show={(stalled || showHint) && !helpOpen}
              text={stalled ? stallText : 'Arrastra la pila para moverla'}
            />
          )}
        </div>

        {ui.showPip && (
          <PipFrame
            layout={pipLayout}
            cameraMode={ui.cameraMode}
            onSwap={() => actions.setCameraMode(ui.cameraMode === 'pov' ? 'overview' : 'pov')}
            onClose={actions.togglePip}
          />
        )}

        <div className={`absolute inset-x-0 flex flex-col items-center gap-3 ${compact ? 'bottom-3' : 'bottom-5'}`}>
          {!compact && (
            <Hint
              show={(stalled || showHint) && !helpOpen}
              text={
                stalled
                  ? stallText
                  : 'Arrastra la pila para moverla o lanzarla · Pulsa ? para ver los atajos'
              }
            />
          )}
          <Dock
            ui={ui}
            actions={actions}
            compact={compact}
            wide={wide}
            onOpenHelp={() => {
              setShowHint(false);
              setHelpOpen(true);
            }}
          />
        </div>

        <HelpSheet open={helpOpen} onClose={() => setHelpOpen(false)} />
      </div>
    </MotionConfig>
  );
}
