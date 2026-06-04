import {makeScene2D, Rect, Txt, Video, Img} from '@motion-canvas/2d';
import {createRef, Reference, all, waitFor, linear} from '@motion-canvas/core';
import {BLUE, GREY, DARK, BG} from '../colors';

// Side-by-side comparison of one robot (workstation11) under two configs:
// STARVED (dynamic_50) vs NOT STARVED (dynamic_1). Each video plays with a live
// replay of its actions-left heatmap revealing left-to-right underneath.

const VIEW_W = 1950; // 13in @ 150 dpi
const VIEW_H = 1125; // 7.5in @ 150 dpi

const VIDEO_W = 840;
const VIDEO_H = (VIDEO_W * 9) / 16; // 472.5, 16:9
const HM_W = VIDEO_W;
const HM_H = 96;
const COL_DX = 492;

const VIDEO_CY = -34;
const LABEL_Y = VIDEO_CY - VIDEO_H / 2 - 56;
const HM_CY = VIDEO_CY + VIDEO_H / 2 + 26 + HM_H / 2;
const CAP_Y = HM_CY + HM_H / 2 + 34;

const PLAY_DUR = 27;

interface Panel {
  video: Reference<Video>;
  cover: Reference<Rect>;
  cursor: Reference<Rect>;
}

export default makeScene2D(function* (view) {
  view.size(VIEW_W, VIEW_H);
  view.fill(BG);

  function buildPanel(
    cx: number,
    videoSrc: string,
    heatmapSrc: string,
    labelText: string,
    accent: string,
  ): Panel {
    const video = createRef<Video>();
    const cover = createRef<Rect>();
    const cursor = createRef<Rect>();

    view.add(
      <Txt
        position={[cx, LABEL_Y]}
        text={labelText}
        fontFamily={'Helvetica Neue'}
        fontWeight={700}
        fontSize={58}
        fill={accent}
      />,
    );
    view.add(
      <Rect
        position={[cx, VIDEO_CY]}
        size={[VIDEO_W, VIDEO_H]}
        radius={18}
        stroke={accent}
        lineWidth={6}
        fill={'#000000'}
        clip
      >
        <Video ref={video} src={videoSrc} size={[VIDEO_W, VIDEO_H]} />
      </Rect>,
    );
    // Heatmap strip with a left-to-right reveal (cover shrinks; cursor tracks).
    view.add(
      <Rect
        position={[cx, HM_CY]}
        size={[HM_W, HM_H]}
        radius={12}
        stroke={accent}
        lineWidth={4}
        fill={'#FFFFFF'}
        clip
      >
        <Img position={[0, 0]} size={[HM_W, HM_H]} src={heatmapSrc} smoothing={false} />
        <Rect
          ref={cover}
          position={[HM_W / 2, 0]}
          offset={[1, 0]}
          size={[HM_W, HM_H]}
          fill={BG}
        />
        <Rect ref={cursor} position={[-HM_W / 2, 0]} size={[4, HM_H]} fill={DARK} />
      </Rect>,
    );
    view.add(
      <Txt
        position={[cx, CAP_Y]}
        text={'Executed action chunks over time'}
        fontFamily={'Helvetica Neue'}
        fontWeight={500}
        fontSize={30}
        fill={DARK}
      />,
    );
    return {video, cover, cursor};
  }

  const panels: Panel[] = [
    buildPanel(-COL_DX, '/starved.mp4', '/heatmaps/starved.png', 'STARVED', GREY),
    buildPanel(COL_DX, '/notstarved.mp4', '/heatmaps/notstarved.png', 'NOT STARVED', BLUE),
  ];

  // Start both videos together, then reveal both heatmaps in lockstep.
  for (const p of panels) {
    p.video().play();
  }
  yield* all(
    ...panels.flatMap((p) => [
      p.cover().size.x(0, PLAY_DUR, linear),
      p.cursor().position.x(HM_W / 2, PLAY_DUR, linear),
    ]),
  );
  yield* waitFor(1.0);
});
