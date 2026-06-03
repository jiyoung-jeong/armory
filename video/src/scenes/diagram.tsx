import {makeScene2D, Rect, Txt, Line} from '@motion-canvas/2d';
import {
  createRef,
  all,
  waitFor,
  delay,
  easeInOutCubic,
} from '@motion-canvas/core';

import {Robot} from '../components/Robot';
import {Server} from '../components/Server';
import {BLUE, GREY, BG, withAlpha} from '../colors';

// All positions below are in view-centered coords matching the source Figma
// canvas (3722 x 854). Figma point (fx, fy) maps to (fx - 1861, fy - 427).
const ROBOT1_POS: [number, number] = [-1009, -209];
const ROBOTN_POS: [number, number] = [-1009, 262];
const SERVER_POS: [number, number] = [1011.5, 0];

// Arrow endpoints across the gap between robots and server.
const R1_OBS_ARROW: [[number, number], [number, number]] = [
  [-102, -272.3],
  [112.5, -272.3],
];
const R1_ACT_ARROW: [[number, number], [number, number]] = [
  [-102, -150.3],
  [113, -150.3],
];
const RN_OBS_ARROW: [[number, number], [number, number]] = [
  [-102, 198.7],
  [112.5, 198.7],
];
const RN_ACT_ARROW: [[number, number], [number, number]] = [
  [-102, 320.7],
  [113, 320.7],
];

export default makeScene2D(function* (view) {
  view.size(3722, 854);
  view.fill(BG);

  const robot1 = createRef<Robot>();
  const robotN = createRef<Robot>();
  const server = createRef<Server>();

  const r1ObsMsg = createRef<Rect>();
  const r1ActMsg = createRef<Rect>();
  const rnObsMsg = createRef<Rect>();
  const rnActMsg = createRef<Rect>();

  view.add(
    <>
      <Robot
        ref={robot1}
        label="Robot 1"
        color={BLUE}
        labelBg={withAlpha(BLUE, 0.6)}
        maxSlots={4}
        initialFill={4}
        decay={0.22}
        position={ROBOT1_POS}
      />
      <Robot
        ref={robotN}
        label="Robot N"
        color={GREY}
        labelBg={withAlpha(GREY, 0.6)}
        maxSlots={7}
        initialFill={7}
        decay={0.15}
        position={ROBOTN_POS}
      />
      <Server ref={server} position={SERVER_POS} />

      {/* Static arrows. Top per robot = observation (robot → server),
        * bottom = action (server → robot). */}
      <Line points={R1_OBS_ARROW} stroke="#000" lineWidth={10} endArrow arrowSize={22} />
      <Line points={R1_ACT_ARROW} stroke="#000" lineWidth={10} startArrow arrowSize={22} />
      <Line points={RN_OBS_ARROW} stroke="#000" lineWidth={10} endArrow arrowSize={22} />
      <Line points={RN_ACT_ARROW} stroke="#000" lineWidth={10} startArrow arrowSize={22} />

      {/* Static labels for the arrows. */}
      <Txt position={[5, -325]} text="oₜ" fontFamily="Helvetica Neue" fontSize={60} fontWeight={700} fill="#000" />
      <Txt position={[5, -200]} text="Aₜ" fontFamily="Helvetica Neue" fontSize={60} fontWeight={700} fill="#000" />
      <Txt position={[5, 148]} text="oₜ" fontFamily="Helvetica Neue" fontSize={60} fontWeight={700} fill="#000" />
      <Txt position={[5, 273]} text="Aₜ" fontFamily="Helvetica Neue" fontSize={60} fontWeight={700} fill="#000" />

      {/* Messenger dots — fly along the arrows during each step. */}
      <Rect ref={r1ObsMsg} size={36} radius={18} fill={BLUE} opacity={0} />
      <Rect ref={r1ActMsg} size={36} radius={18} fill={BLUE} opacity={0} />
      <Rect ref={rnObsMsg} size={36} radius={18} fill={GREY} opacity={0} />
      <Rect ref={rnActMsg} size={36} radius={18} fill={GREY} opacity={0} />
    </>,
  );

  function* flyMessenger(
    msg: Rect,
    arrow: [[number, number], [number, number]],
    direction: 'forward' | 'reverse',
    duration = 0.6,
  ) {
    const [start, end] = direction === 'forward' ? arrow : [arrow[1], arrow[0]];
    msg.position(start);
    yield* msg.opacity(1, 0.1);
    yield* msg.position(end, duration, easeInOutCubic);
    yield* msg.opacity(0, 0.1);
  }

  // Send a fresh chunk back every CHUNK_PERIOD steps. Robots send observations
  // and consume one action every step.
  const NUM_STEPS = 9;
  const CHUNK_PERIOD = 3;

  // Hold on the initial state briefly.
  yield* waitFor(0.6);

  for (let step = 0; step < NUM_STEPS; step++) {
    // Robots send observations and exhaust one action each (queue shifts left).
    yield* all(
      flyMessenger(r1ObsMsg(), R1_OBS_ARROW, 'forward'),
      flyMessenger(rnObsMsg(), RN_OBS_ARROW, 'forward'),
      delay(0.15, robot1().consumeAction(0.5)),
      delay(0.15, robotN().consumeAction(0.5)),
    );

    yield* waitFor(0.25);

    // Server returns action chunks only every CHUNK_PERIOD steps.
    if ((step + 1) % CHUNK_PERIOD === 0) {
      yield* all(
        flyMessenger(r1ActMsg(), R1_ACT_ARROW, 'reverse'),
        flyMessenger(rnActMsg(), RN_ACT_ARROW, 'reverse'),
        delay(0.55, robot1().receiveActionToFull(0.6)),
        delay(0.55, robotN().receiveActionToFull(0.6)),
      );
    }

    yield* waitFor(0.3);
  }
});
