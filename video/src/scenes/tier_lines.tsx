import {makeScene2D, Rect, Txt, Line, Layout, Circle} from '@motion-canvas/2d';
import {
  createRef,
  Reference,
  ThreadGenerator,
  all,
  waitFor,
  delay,
  easeInOutCubic,
} from '@motion-canvas/core';
import {DARK, BG, withAlpha} from '../colors';
import {ROBOTS, ONE_FAST, HALF_FAST, Scenario} from './tier_data';

// Sim tier-breakdown as animated line charts. For each scenario (One Fast, then
// Half Fast) a 2x2 grid: columns = Fast tier / Slow tier, rows = throughput /
// starvation vs number of robots, one line per scheduler.

const VIEW_W = 1950;
const VIEW_H = 1125;

const SCHED: {name: string; color: string}[] = [
  {name: 'EDF', color: '#8E6CA8'},
  {name: 'RR', color: '#5FA86F'},
  {name: 'LA', color: '#6FB0D6'},
  {name: 'LA@3', color: '#3C86B8'},
  {name: 'LA@5', color: '#1E5C84'},
];

// Panel geometry
const PW = 700;
const PH = 358;
const FAST_CX = -420;
const SLOW_CX = 420;
const THR_CY = -110;
const STARV_CY = 300;
const ML = 84; // left margin (y labels)
const MR = 26;
const MT = 24;
const MB = 44; // bottom margin (x labels)

interface Region {
  x0: number;
  x1: number;
  yb: number;
  yt: number;
}
const region = (cx: number, cy: number): Region => ({
  x0: cx - PW / 2 + ML,
  x1: cx + PW / 2 - MR,
  yb: cy + PH / 2 - MB,
  yt: cy - PH / 2 + MT,
});
const xMap = (r: Region, i: number) => r.x0 + ((r.x1 - r.x0) * i) / (ROBOTS.length - 1);
const yMap = (r: Region, v: number, lo: number, hi: number) => {
  const y = r.yb - ((v - lo) / (hi - lo)) * (r.yb - r.yt);
  return Math.max(r.yt, Math.min(r.yb, y)); // clamp to the panel
};

interface PanelCfg {
  cx: number;
  cy: number;
  forceZero: boolean; // starvation panels keep 0 as the floor
  series: Record<string, number[]>;
  showY: boolean;
  showX: boolean;
  yLabel?: string;
}

// Tight y-axis fitted to a panel's own data, with a "nice" tick step.
function niceAxis(series: Record<string, number[]>, forceZero: boolean) {
  let mn = Infinity;
  let mx = -Infinity;
  for (const arr of Object.values(series)) {
    for (const x of arr) {
      if (x < mn) mn = x;
      if (x > mx) mx = x;
    }
  }
  if (forceZero) mn = 0;
  const span = mx - mn || 1;
  const raw = span / 4;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 2.5 ? 2.5 : norm <= 5 ? 5 : 10) * mag;
  const lo = forceZero ? 0 : Math.floor(mn / step) * step;
  let hi = Math.ceil(mx / step) * step;
  if (hi - mx < step * 0.25) hi += step; // headroom so the peak isn't on the edge
  const ticks: number[] = [];
  for (let t = lo; t <= hi + 1e-9; t += step) ticks.push(Math.round(t * 100) / 100);
  return {lo, hi, ticks};
}

function addPanel(
  parent: Layout,
  cfg: PanelCfg,
  lineRefs: Reference<Line>[],
  dotRefs: Reference<Layout>[],
) {
  const r = region(cfg.cx, cfg.cy);
  const {lo, hi, ticks} = niceAxis(cfg.series, cfg.forceZero);

  // gridlines + ticks + axes
  parent.add(
    <Layout layout={false}>
      {ticks.map((t) => (
        <Layout>
          <Line points={[[r.x0, yMap(r, t, lo, hi)], [r.x1, yMap(r, t, lo, hi)]]} stroke={withAlpha(DARK, 0.14)} lineWidth={2} />
          {cfg.showY && (
            <Txt position={[r.x0 - 16, yMap(r, t, lo, hi)]} offset={[1, 0]} text={`${t}`} fontFamily={'Helvetica Neue'} fontSize={26} fill={DARK} />
          )}
        </Layout>
      ))}
      <Line points={[[r.x0, r.yt], [r.x0, r.yb]]} stroke={DARK} lineWidth={3} />
      <Line points={[[r.x0, r.yb], [r.x1, r.yb]]} stroke={DARK} lineWidth={3} />
      {cfg.showX &&
        ROBOTS.map((rob, i) => (
          <Txt position={[xMap(r, i), r.yb + 30]} text={`${rob}`} fontFamily={'Helvetica Neue'} fontSize={26} fill={DARK} />
        ))}
      {cfg.yLabel && (
        <Txt position={[r.x0 - 94, (r.yt + r.yb) / 2]} rotation={-90} text={cfg.yLabel} fontFamily={'Helvetica Neue'} fontWeight={500} fontSize={28} fill={DARK} />
      )}
    </Layout>,
  );

  // series: a line (drawn via end 0->1) and a dot layer (faded in)
  for (const s of SCHED) {
    const vals = cfg.series[s.name];
    if (!vals) continue;
    const pts = vals.map((v, i) => [xMap(r, i), yMap(r, v, lo, hi)] as [number, number]);
    const lr = createRef<Line>();
    const dr = createRef<Layout>();
    lineRefs.push(lr);
    dotRefs.push(dr);
    parent.add(<Line ref={lr} points={pts} stroke={s.color} lineWidth={4} end={0} lineJoin={'round'} />);
    parent.add(
      <Layout ref={dr} layout={false} opacity={0}>
        {pts.map((p) => (
          <Circle position={p} size={14} fill={s.color} stroke={BG} lineWidth={2} />
        ))}
      </Layout>,
    );
  }
}

interface Built {
  container: Reference<Layout>;
  animateIn: () => ThreadGenerator;
}

function buildScenario(view: any, sc: Scenario): Built {
  const container = createRef<Layout>();
  const lineRefs: Reference<Line>[] = [];
  const dotRefs: Reference<Layout>[] = [];

  const root = (
    <Layout ref={container} layout={false} opacity={0}>
      <Txt position={[0, -512]} text={sc.title} fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={52} fill={DARK} />
      <Txt position={[0, -464]} text={sc.subtitle} fontFamily={'Helvetica Neue'} fontWeight={500} fontSize={30} fill={withAlpha(DARK, 0.7)} />
      {/* legend */}
      {SCHED.map((s, i) => (
        <Layout position={[-300 + i * 150, -418]} layout={false}>
          <Line points={[[-26, 0], [10, 0]]} stroke={s.color} lineWidth={5} />
          <Circle position={[-8, 0]} size={13} fill={s.color} stroke={BG} lineWidth={2} />
          <Txt position={[20, 0]} offset={[-1, 0]} text={s.name} fontFamily={'Helvetica Neue'} fontWeight={600} fontSize={28} fill={DARK} />
        </Layout>
      ))}
      {/* column headers */}
      <Txt position={[FAST_CX, -344]} text={'Fast tier'} fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={38} fill={DARK} />
      <Txt position={[SLOW_CX, -344]} text={'Slow tier'} fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={38} fill={DARK} />
      {/* x-axis title */}
      <Txt position={[0, STARV_CY + PH / 2 + 30]} text={'Number of robots'} fontFamily={'Helvetica Neue'} fontWeight={500} fontSize={30} fill={DARK} />
    </Layout>
  ) as Layout;
  view.add(root);

  addPanel(container(), {cx: FAST_CX, cy: THR_CY, forceZero: false, series: sc.series.fastThr, showY: true, showX: false, yLabel: 'Throughput (succ / min)'}, lineRefs, dotRefs);
  addPanel(container(), {cx: SLOW_CX, cy: THR_CY, forceZero: false, series: sc.series.slowThr, showY: true, showX: false}, lineRefs, dotRefs);
  addPanel(container(), {cx: FAST_CX, cy: STARV_CY, forceZero: true, series: sc.series.fastStarv, showY: true, showX: true, yLabel: 'Starvation (%)'}, lineRefs, dotRefs);
  addPanel(container(), {cx: SLOW_CX, cy: STARV_CY, forceZero: true, series: sc.series.slowStarv, showY: true, showX: true}, lineRefs, dotRefs);

  function* animateIn(): ThreadGenerator {
    yield* container().opacity(1, 0.6, easeInOutCubic); // reveal chrome (lines still hidden)
    yield* all(...lineRefs.map((l, i) => delay((i % 5) * 0.05, l().end(1, 1.3, easeInOutCubic))));
    yield* all(...dotRefs.map((d) => d().opacity(1, 0.4)));
  }

  return {container, animateIn};
}

export default makeScene2D(function* (view) {
  view.size(VIEW_W, VIEW_H);
  view.fill(BG);

  const c1 = buildScenario(view, ONE_FAST);
  yield* c1.animateIn();
  yield* waitFor(8);
  yield* c1.container().opacity(0, 0.6, easeInOutCubic);

  const c2 = buildScenario(view, HALF_FAST);
  yield* c2.animateIn();
  yield* waitFor(8);
});
