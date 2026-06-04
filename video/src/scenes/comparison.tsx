import {makeScene2D, Rect, Txt, Video, Layout} from '@motion-canvas/2d';
import {
  createRef,
  Reference,
  all,
  waitFor,
  easeInOutCubic,
} from '@motion-canvas/core';
import {BLUE, GREY, DARK, BG, withAlpha} from '../colors';

// Two side-by-side grids of the same 10 real-robot videos, one per scheduler.
// All videos start at t=0 and play together; around the midpoint we zoom each
// grid in on workstation11 (the fast robot) for a side-by-side comparison.

const ORDER = [
  'workstation1',
  'workstation4',
  'workstation5',
  'workstation6',
  'workstation7',
  'workstation8',
  'workstation11',
  'workstation12',
  'workstation13',
  'workstation14',
];
const ZOOM_TARGET = 'workstation11';

const COLS = 2;
const ROWS = 5;
const CELL_W = 384;
const CELL_H = (CELL_W * 9) / 16; // 216, 16:9, matches the 1280x720 source
const GAP = 14;
const CELL_RADIUS = 14;

const GRID_W = COLS * CELL_W + (COLS - 1) * GAP;
const GRID_H = ROWS * CELL_H + (ROWS - 1) * GAP;

const FRAME_PAD = 12; // white inset between the grid and the rounded border
const FRAME_RADIUS = 28;
const FRAME_W = GRID_W + 2 * FRAME_PAD;
const FRAME_H = GRID_H + 2 * FRAME_PAD;

const MARGIN = 60; // outer breathing room so nothing is cut off
const LABEL_H = 100; // band above each grid for its label
const CENTER_GAP = 110; // horizontal spacing between the two grids

const VIEW_W = 2 * FRAME_W + CENTER_GAP + 2 * MARGIN;
const VIEW_H = MARGIN + LABEL_H + FRAME_H + MARGIN;
const GRIDS_CY = -VIEW_H / 2 + MARGIN + LABEL_H + FRAME_H / 2;
const LEFT_CX = -(FRAME_W + CENTER_GAP) / 2;
const RIGHT_CX = (FRAME_W + CENTER_GAP) / 2;

// Zoom scale so the target cell fills the grid (frame interior) width.
const ZOOM_S = GRID_W / CELL_W;
const FRAME_ZOOM_H = CELL_H * ZOOM_S + 2 * FRAME_PAD;
const LABEL_ORIG_Y = GRIDS_CY - FRAME_H / 2 - LABEL_H / 2;
const LABEL_ZOOM_Y = GRIDS_CY - FRAME_ZOOM_H / 2 - LABEL_H / 2;
const HEADER_Y = LABEL_ZOOM_Y - 135; // caption above the method labels when zoomed
const FOOTER_Y = GRIDS_CY + FRAME_ZOOM_H / 2 + 130; // caption below zoomed videos
const ZOOM_DUR = 1.5;

function cellPos(i: number): [number, number] {
  const col = i % COLS;
  const row = Math.floor(i / COLS);
  return [
    -GRID_W / 2 + CELL_W / 2 + col * (CELL_W + GAP),
    -GRID_H / 2 + CELL_H / 2 + row * (CELL_H + GAP),
  ];
}

const TARGET_POS = cellPos(ORDER.indexOf(ZOOM_TARGET));
// A non-target (quasi-static / slow) robot to highlight before workstation11.
const QUASI_TARGET = 'workstation5';
const QUASI_POS = cellPos(ORDER.indexOf(QUASI_TARGET));

interface Half {
  frame: Reference<Rect>;
  grid: Reference<Layout>;
  label: Reference<Txt>;
  videos: Reference<Video>[];
  initialText: string;
  zoomText: string;
}

// Cross-fade a label to new text.
function* transformLabel(label: Txt, newText: string) {
  yield* label.opacity(0, 0.4, easeInOutCubic);
  label.text(newText);
  yield* label.opacity(1, 0.4, easeInOutCubic);
}

// Zoom a grid onto a target cell and collapse its frame to that single video.
function* zoomInto(h: Half, pos: [number, number], dur: number) {
  yield* all(
    h.grid().scale(ZOOM_S, dur, easeInOutCubic),
    h.grid().position([-ZOOM_S * pos[0], -ZOOM_S * pos[1]], dur, easeInOutCubic),
    h.frame().size([FRAME_W, FRAME_ZOOM_H], dur, easeInOutCubic),
  );
}

// Restore a grid to the full-grid view.
function* zoomReset(h: Half, dur: number) {
  yield* all(
    h.grid().scale(1, dur, easeInOutCubic),
    h.grid().position([0, 0], dur, easeInOutCubic),
    h.frame().size([FRAME_W, FRAME_H], dur, easeInOutCubic),
  );
}

// Reveal the surviving grid as the engine: cross-fade the label to "Armory",
// enlarged and recolored.
function* revealArmory(label: Txt) {
  yield* label.opacity(0, 0.4, easeInOutCubic);
  label.text('Armory');
  label.fontSize(96);
  label.fill(BLUE);
  yield* label.opacity(1, 0.5, easeInOutCubic);
}

export default makeScene2D(function* (view) {
  view.size(VIEW_W, VIEW_H);
  view.fill(BG);

  function buildHalf(
    centerX: number,
    base: string,
    labelText: string,
    zoomText: string,
    accent: string,
  ): Half {
    const frame = createRef<Rect>();
    const grid = createRef<Layout>();
    const label = createRef<Txt>();
    const videos = ORDER.map(() => createRef<Video>());

    view.add(
      <Rect
        ref={frame}
        position={[centerX, GRIDS_CY]}
        size={[FRAME_W, FRAME_H]}
        fill={'#FFFFFF'}
        stroke={accent}
        lineWidth={6}
        radius={FRAME_RADIUS}
        clip
      >
        <Layout ref={grid} layout={false}>
          {ORDER.map((ws, i) => (
            <Rect
              position={cellPos(i)}
              size={[CELL_W, CELL_H]}
              radius={CELL_RADIUS}
              fill={'#000000'}
              clip
            >
              <Video
                ref={videos[i]}
                src={`/${base}/${ws}.mp4`}
                size={[CELL_W, CELL_H]}
              />
              <Rect
                offset={[-1, -1]}
                position={[-CELL_W / 2 + 12, -CELL_H / 2 + 12]}
                fill={withAlpha(DARK, 0.62)}
                radius={9}
                layout
                padding={[6, 13]}
              >
                <Txt
                  text={`Robot #${ws.replace('workstation', '')}`}
                  fontFamily={'Helvetica Neue'}
                  fontWeight={600}
                  fontSize={24}
                  fill={'#FFFFFF'}
                />
              </Rect>
            </Rect>
          ))}
        </Layout>
      </Rect>,
    );
    // Label in the band above each grid.
    view.add(
      <Txt
        ref={label}
        position={[centerX, GRIDS_CY - FRAME_H / 2 - LABEL_H / 2]}
        text={labelText}
        fontFamily={'Helvetica Neue'}
        fontWeight={700}
        fontSize={60}
        fill={DARK}
      />,
    );
    return {frame, grid, label, videos, initialText: labelText, zoomText};
  }

  const halves: Half[] = [
    buildHalf(
      LEFT_CX,
      'maxbatch_proxy',
      'Naive Scheduling',
      'EDF (Baseline)',
      GREY,
    ),
    buildHalf(
      RIGHT_CX,
      'lookahead_proxy',
      'Smart Scheduling',
      'Lookahead (Ours)',
      BLUE,
    ),
  ];

  // Overlay text: a centered intro statement and a top header for the zooms.
  const intro = createRef<Txt>();
  const header = createRef<Txt>();
  view.add(
    <Txt
      ref={intro}
      position={[0, 0]}
      width={1500}
      text={
        'We introduce a serving system that enables a single ' +
        'GPU to serve many robots through batched inference in a throughput-aware manner.'
      }
      fontFamily={'Helvetica Neue'}
      fontWeight={600}
      fontSize={72}
      fill={DARK}
      textAlign={'center'}
      textWrap
      opacity={0}
    />,
  );
  view.add(
    <Txt
      ref={header}
      position={[0, HEADER_Y]}
      width={1720}
      text={''}
      fontFamily={'Helvetica Neue'}
      fontWeight={600}
      fontSize={48}
      fill={DARK}
      textAlign={'center'}
      textWrap
      opacity={0}
    />,
  );

  // Footer caption shown below the videos during the workstation11 zoom.
  const footer = createRef<Txt>();
  view.add(
    <Txt
      ref={footer}
      position={[0, FOOTER_Y]}
      width={1720}
      text={
        'When fast and slow robots share a GPU, we devise a scheduler that can ' +
        'choose between maximizing average service or protecting robots that ' +
        'exhaust chunks faster.'
      }
      fontFamily={'Helvetica Neue'}
      fontWeight={600}
      fontSize={44}
      fill={DARK}
      textAlign={'center'}
      textWrap
      opacity={0}
    />,
  );

  // Full-screen overlay used to fade the whole scene out at the end.
  const fadeOverlay = createRef<Rect>();
  view.add(
    <Rect ref={fadeOverlay} size={[VIEW_W, VIEW_H]} fill={BG} opacity={0} />,
  );

  // ---- Intro: grids hidden, centered statement on screen ----
  for (const h of halves) {
    h.frame().scale(0);
    h.label().opacity(0);
  }
  yield* intro().opacity(1, 0.6, easeInOutCubic);
  yield* waitFor(3.6);
  yield* intro().opacity(0, 0.6, easeInOutCubic);

  // ---- Grids expand; start playback together so they stay in lockstep ----
  for (const h of halves) {
    for (const v of h.videos) {
      v().play();
    }
  }
  yield* all(
    ...halves.flatMap((h) => [
      h.frame().scale(1, 1.0, easeInOutCubic),
      h.label().opacity(1, 0.9, easeInOutCubic),
    ]),
  );
  yield* waitFor(5.5);

  // ---- Zoom 1: a quasi-static (slow) robot. No method labels here; the
  // header fades in only after the zoom settles so it doesn't overlap the grid.
  yield* all(
    ...halves.map((h) => zoomInto(h, QUASI_POS, ZOOM_DUR)),
    ...halves.map((h) => h.label().opacity(0, 0.6, easeInOutCubic)),
  );
  header().text(
    'Some robots perform quasi-static tasks that are not latency-sensitive.',
  );
  yield* waitFor(0.5);
  yield* header().opacity(1, 0.5, easeInOutCubic);
  yield* waitFor(4.5);

  // Zoom back out; restore the Naive / Smart labels over the full grids.
  yield* header().opacity(0, 0.4, easeInOutCubic);
  yield* all(
    ...halves.map((h) => zoomReset(h, ZOOM_DUR)),
    ...halves.map((h) => h.label().opacity(1, 0.8, easeInOutCubic)),
  );
  yield* waitFor(2.0);

  // ---- Zoom 2: workstation11, the dynamic / latency-sensitive robot. The
  // method labels move up and transform; the header fades in after the zoom.
  yield* all(
    ...halves.flatMap((h) => [
      zoomInto(h, TARGET_POS, ZOOM_DUR),
      h.label().position.y(LABEL_ZOOM_Y, ZOOM_DUR, easeInOutCubic),
      transformLabel(h.label(), h.zoomText),
    ]),
  );
  header().text(
    'Other robots perform highly dynamic, latency-sensitive tasks that may require better service in order to maintain/improve throughput.',
  );
  yield* waitFor(0.5);
  yield* all(
    header().opacity(1, 0.5, easeInOutCubic),
    footer().opacity(1, 0.5, easeInOutCubic),
  );
  yield* waitFor(10.0);

  // ---- Ending: the naive (gray) grid exits to the left while the smart
  // (blue) grid un-zooms, slides to center, and is revealed as "Armory". ----
  yield* all(
    header().opacity(0, 0.4, easeInOutCubic),
    footer().opacity(0, 0.4, easeInOutCubic),
  );

  const gray = halves[0];
  const blue = halves[1];
  const OFF_LEFT = -(VIEW_W / 2 + FRAME_W);

  yield* all(
    // Naive grid un-zooms and slides off the left edge.
    zoomReset(gray, ZOOM_DUR),
    gray.frame().position.x(OFF_LEFT, ZOOM_DUR, easeInOutCubic),
    gray.label().position([OFF_LEFT, LABEL_ORIG_Y], ZOOM_DUR, easeInOutCubic),
    // Smart grid un-zooms and centers.
    zoomReset(blue, ZOOM_DUR),
    blue.frame().position.x(0, ZOOM_DUR, easeInOutCubic),
    blue.label().position([0, LABEL_ORIG_Y - 20], ZOOM_DUR, easeInOutCubic),
    revealArmory(blue.label()),
  );
  yield* waitFor(5.0);

  // ---- Fade the whole scene out over the final seconds ----
  yield* fadeOverlay().opacity(1, 2.5, easeInOutCubic);
});
