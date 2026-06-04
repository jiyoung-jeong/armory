import {makeScene2D, Rect, Txt, Line, Layout} from '@motion-canvas/2d';
import {
  createRef,
  Reference,
  ThreadGenerator,
  all,
  chain,
  waitFor,
  delay,
  easeOutCubic,
  easeInOutCubic,
} from '@motion-canvas/core';
import {BLUE, GREY, DARK, BG, withAlpha} from '../colors';

// Real-world Lego throughput as stacked bars: system throughput split into
// fast-robot (blue) and slow-robot (gray) contributions, with per-trial-total
// error bars. Two charts play in sequence:
//   1) Half Fast — total maintained, throughput reallocated to the fast tier.
//   2) One Fast  — fast tier doubles and the system total rises, for free.

const VIEW_W = 1950;
const VIEW_H = 1125;

interface Bar {
  label: string;
  fast: number;
  slow: number;
  std: number; // std of the total across the 3 trials
}
interface Callout {
  text: string;
  color: string;
  arrowTo: [number, number]; // padded point just to the right of the target bar
}
interface ChartCfg {
  title: string;
  subtitle: string;
  data: Bar[];
  yMax: number;
  yTicks: number[];
  ref: {value: number; label: string};
  callout: Callout;
  caption: string;
}

// Shared geometry (bar layout is identical across charts; only the y scale
// differs by yMax).
const BASE_Y = 366;
const CHART_TOP_Y = -294;
const BAR_W = 180;
const BAR_DX = 300;
const N = 4;
const firstX = -((N - 1) / 2) * BAR_DX;
const xFor = (i: number) => firstX + i * BAR_DX;
const AXIS_X = firstX - BAR_W / 2 - 72;
const RIGHT_X = xFor(N - 1) + BAR_W / 2 + 20;

interface Chart {
  container: Reference<Layout>;
  animateIn: () => ThreadGenerator;
}

function buildChart(view: any, cfg: ChartCfg): Chart {
  const PX = (BASE_Y - CHART_TOP_Y) / cfg.yMax;
  const yOf = (v: number) => BASE_Y - v * PX;

  const container = createRef<Layout>();
  const intro = createRef<Layout>();
  const slowRefs = cfg.data.map(() => createRef<Rect>());
  const fastRefs = cfg.data.map(() => createRef<Rect>());
  const slowLbl = cfg.data.map(() => createRef<Txt>());
  const fastLbl = cfg.data.map(() => createRef<Txt>());
  const totalLbl = cfg.data.map(() => createRef<Txt>());
  const errBars = cfg.data.map(() => createRef<Layout>());
  const line = createRef<Line>();
  const lineLabel = createRef<Txt>();
  const callout = createRef<Layout>();
  const caption = createRef<Txt>();

  const legend = (label: string, color: string, lx: number) => (
    <Layout position={[lx, -366]} layout={false}>
      <Rect size={[32, 32]} radius={6} fill={color} />
      <Txt position={[26, 0]} offset={[-1, 0]} text={label} fontFamily={'Helvetica Neue'} fontWeight={500} fontSize={30} fill={DARK} />
    </Layout>
  );

  view.add(
    <Layout ref={container} layout={false}>
      <Txt position={[0, -474]} text={cfg.title} fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={52} fill={DARK} />
      <Txt position={[0, -420]} text={cfg.subtitle} fontFamily={'Helvetica Neue'} fontWeight={500} fontSize={32} fill={withAlpha(DARK, 0.7)} />
      {legend('Fast robots', BLUE, -150)}
      {legend('Slow robots', GREY, 110)}

      {/* axes / gridlines / labels */}
      <Layout ref={intro} layout={false} opacity={0}>
        <Line points={[[AXIS_X, CHART_TOP_Y], [AXIS_X, BASE_Y]]} stroke={DARK} lineWidth={3} />
        {cfg.yTicks.map((v) => (
          <Layout>
            <Line points={[[AXIS_X, yOf(v)], [RIGHT_X, yOf(v)]]} stroke={withAlpha(DARK, v === 0 ? 0.55 : 0.16)} lineWidth={v === 0 ? 3 : 2} />
            <Txt position={[AXIS_X - 22, yOf(v)]} offset={[1, 0]} text={`${v}`} fontFamily={'Helvetica Neue'} fontSize={30} fill={DARK} />
          </Layout>
        ))}
        <Txt position={[AXIS_X - 92, (CHART_TOP_Y + BASE_Y) / 2]} rotation={-90} text={'System throughput (successes / min)'} fontFamily={'Helvetica Neue'} fontWeight={500} fontSize={32} fill={DARK} />
        {cfg.data.map((d, i) => (
          <Txt position={[xFor(i), BASE_Y + 40]} text={d.label} fontFamily={'Helvetica Neue'} fontWeight={600} fontSize={36} fill={DARK} />
        ))}
      </Layout>

      {/* bars + labels + error bars */}
      {cfg.data.map((d, i) => {
        const slowH = d.slow * PX;
        const fastH = d.fast * PX;
        const total = d.fast + d.slow;
        const cap = 24;
        return (
          <Layout layout={false}>
            <Rect ref={slowRefs[i]} offset={[0, 1]} position={[xFor(i), BASE_Y]} size={[BAR_W, 0]} fill={GREY} />
            <Rect ref={fastRefs[i]} offset={[0, 1]} position={[xFor(i), BASE_Y - slowH]} size={[BAR_W, 0]} fill={BLUE} radius={[12, 12, 0, 0]} />
            <Txt ref={slowLbl[i]} position={[xFor(i), BASE_Y - slowH / 2]} text={`${Math.round(d.slow)}`} fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={34} fill={'#FFFFFF'} opacity={0} />
            <Txt ref={fastLbl[i]} position={[xFor(i) - BAR_W / 2 - 18, BASE_Y - slowH - fastH / 2]} offset={[1, 0]} text={`${Math.round(d.fast)}`} fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={34} fill={BLUE} opacity={0} />
            <Txt ref={totalLbl[i]} position={[xFor(i), yOf(total + d.std) - 34]} text={`${Math.round(total)}`} fontFamily={'Helvetica Neue'} fontWeight={700} fontSize={38} fill={DARK} opacity={0} />
            <Layout ref={errBars[i]} layout={false} opacity={0}>
              <Line points={[[xFor(i), yOf(total - d.std)], [xFor(i), yOf(total + d.std)]]} stroke={DARK} lineWidth={3} />
              <Line points={[[xFor(i) - cap, yOf(total + d.std)], [xFor(i) + cap, yOf(total + d.std)]]} stroke={DARK} lineWidth={3} />
              <Line points={[[xFor(i) - cap, yOf(total - d.std)], [xFor(i) + cap, yOf(total - d.std)]]} stroke={DARK} lineWidth={3} />
            </Layout>
          </Layout>
        );
      })}

      {/* reference line */}
      <Line ref={line} points={[[AXIS_X, yOf(cfg.ref.value)], [AXIS_X, yOf(cfg.ref.value)]]} stroke={DARK} lineWidth={3} lineDash={[14, 10]} />
      <Txt ref={lineLabel} position={[RIGHT_X + 14, yOf(cfg.ref.value)]} offset={[-1, 0]} text={cfg.ref.label} fontFamily={'Helvetica Neue'} fontWeight={600} fontSize={28} fill={DARK} opacity={0} />

      {/* callout: short arrow (padded off the bar) + text to its right */}
      <Layout ref={callout} layout={false} opacity={0}>
        <Txt position={[cfg.callout.arrowTo[0] + 92, cfg.callout.arrowTo[1]]} offset={[-1, 0]} text={cfg.callout.text} fontFamily={'Helvetica Neue'} fontWeight={600} fontSize={34} fill={cfg.callout.color} />
        <Line points={[[cfg.callout.arrowTo[0] + 78, cfg.callout.arrowTo[1]], cfg.callout.arrowTo]} stroke={cfg.callout.color} lineWidth={4} endArrow arrowSize={16} />
      </Layout>

      {/* caption */}
      <Txt ref={caption} position={[0, BASE_Y + 132]} width={1480} text={cfg.caption} fontFamily={'Helvetica Neue'} fontWeight={500} fontSize={34} fill={DARK} textAlign={'center'} textWrap opacity={0} />
    </Layout>,
  );

  function* animateIn(): ThreadGenerator {
    yield* intro().opacity(1, 0.6, easeInOutCubic);
    yield* all(
      ...cfg.data.map((d, i) =>
        delay(i * 0.1, chain(slowRefs[i]().size.y(d.slow * PX, 0.6, easeOutCubic), slowLbl[i]().opacity(1, 0.25))),
      ),
    );
    yield* all(
      ...cfg.data.map((d, i) =>
        delay(i * 0.1, chain(fastRefs[i]().size.y(d.fast * PX, 0.6, easeOutCubic), fastLbl[i]().opacity(1, 0.25))),
      ),
    );
    yield* all(
      ...totalLbl.map((t) => t().opacity(1, 0.4)),
      ...errBars.map((e, i) => delay(0.1 * i, e().opacity(1, 0.4))),
    );
    yield* all(
      line().points([[AXIS_X, yOf(cfg.ref.value)], [RIGHT_X, yOf(cfg.ref.value)]], 0.7, easeInOutCubic),
      delay(0.4, lineLabel().opacity(1, 0.4)),
    );
    yield* callout().opacity(1, 0.5, easeInOutCubic);
    yield* caption().opacity(1, 0.6, easeInOutCubic);
  }

  return {container, animateIn};
}

// ---- Per-chart configs ----
const HALF_FAST: ChartCfg = {
  title: 'Better scheduling can reallocate throughput to fast robots.',
  subtitle: 'Real-world robots — Half Fast (5 fast, 5 slow)',
  data: [
    {label: 'EDF', fast: 15.7, slow: 47.0, std: 6.81},
    {label: 'RR', fast: 19.3, slow: 45.3, std: 3.79},
    {label: 'LA', fast: 18.0, slow: 48.3, std: 5.51},
    {label: 'LA@5', fast: 30.7, slow: 32.7, std: 7.23},
  ],
  yMax: 80,
  yTicks: [0, 20, 40, 60, 80],
  ref: {value: 64, label: 'system throughput\nmaintained'},
  callout: {
    text: '~2x EDF',
    color: BLUE,
    // LA@5 fast-tier midpoint at yMax=80, padded just right of the bar
    arrowTo: [xFor(3) + BAR_W / 2 + 18, BASE_Y - (32.7 + 30.7 / 2) * ((BASE_Y - CHART_TOP_Y) / 80)],
  },
  caption:
    'Lookahead trades slow-robot throughput for fast-robot throughput while ' +
    'keeping the system total roughly constant.',
};

const ONE_FAST: ChartCfg = {
  title: 'Better scheduling can also boost fast robots for free.',
  subtitle: 'Real-world robots — One Fast (1 fast, 9 slow)',
  data: [
    {label: 'EDF', fast: 3.7, slow: 79.3, std: 2.65},
    {label: 'RR', fast: 2.0, slow: 81.3, std: 8.02},
    {label: 'LA', fast: 3.7, slow: 88.7, std: 4.04},
    {label: 'LA@5', fast: 8.3, slow: 90.3, std: 5.86},
  ],
  yMax: 120,
  yTicks: [0, 20, 40, 60, 80, 100],
  ref: {value: 83, label: 'EDF baseline'},
  callout: {
    text: '2x EDF, 4x RR',
    color: BLUE,
    // LA@5 fast-tier midpoint at yMax=120, padded just right of the bar
    arrowTo: [xFor(3) + BAR_W / 2 + 18, BASE_Y - (90.3 + 8.3 / 2) * ((BASE_Y - CHART_TOP_Y) / 120)],
  },
  caption:
    'By smartly choosing batches, lookahead doubles fast-robot throughput and ' +
    'even slightly benefits slow robots, lifting the system total above the baselines by 18%.',
};

export default makeScene2D(function* (view) {
  view.size(VIEW_W, VIEW_H);
  view.fill(BG);

  const c1 = buildChart(view, HALF_FAST);
  yield* c1.animateIn();
  yield* waitFor(6);
  yield* c1.container().opacity(0, 0.6, easeInOutCubic);

  const c2 = buildChart(view, ONE_FAST);
  yield* c2.animateIn();
  yield* waitFor(6);
});
