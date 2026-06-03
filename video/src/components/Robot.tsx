import {Layout, LayoutProps, Rect, Txt} from '@motion-canvas/2d';
import {DARK} from '../colors';
import {ActionQueue} from './ActionQueue';

export interface RobotProps extends LayoutProps {
  label: string;
  color: string;
  labelBg: string;
  maxSlots: number;
  initialFill: number;
  decay?: number;
  screenshotChildren?: any;
}

// Dimensions taken from the Figma export.
export const ROBOT_PANEL_W = 1408;
export const ROBOT_PANEL_H = 312;
// Screenshot placeholder temporarily hidden. Keep the dimensions nearby so the
// right-side box can be restored without remeasuring the layout.
// export const ROBOT_SCREENSHOT_W = 272;
// export const ROBOT_SCREENSHOT_H = 272;
// export const ROBOT_SCREENSHOT_GAP = 24;
export const ROBOT_TOTAL_W = ROBOT_PANEL_W;
export const ROBOT_TOTAL_H = ROBOT_PANEL_H;

// Inside the panel (panel-local center origin):
const PANEL_LOCAL = {
  // "Robot N" label box: Figma (66, 158.7) within panel at (0, 62)
  labelCenter: [-488, -2.8] as [number, number],
  labelSize: [300, 113] as [number, number],
  // "actions left" text region: Figma (436, 126)
  actionsLabelCenter: [-169, -3] as [number, number],
  actionsLabelSize: [198, 178] as [number, number],
  // Queue panel background box: Figma (436, 126) w=908, h=178
  queuePanelCenter: [186, -3] as [number, number],
  queuePanelSize: [908, 178] as [number, number],
  // Chunk 0 center: Figma (688+30, 185+30)
  queueChunk0Center: [14, -3] as [number, number],
};

// Panel center within Robot Layout's coords.
const PANEL_OFFSET_X = -ROBOT_TOTAL_W / 2 + ROBOT_PANEL_W / 2;
// const SCREENSHOT_OFFSET_X = ROBOT_TOTAL_W / 2 - ROBOT_SCREENSHOT_W / 2;
const PANEL_RIGHT_X = PANEL_OFFSET_X + ROBOT_PANEL_W / 2;

function robotLocal(panelLocal: [number, number]): [number, number] {
  return [PANEL_OFFSET_X + panelLocal[0], panelLocal[1]];
}

export class Robot extends Layout {
  public readonly queue: ActionQueue;
  // public readonly screenshot: Rect;

  // Anchors (Robot-local coords) for arrows in/out of this robot.
  // Top anchor — observation going to server.
  // Bottom anchor — action coming from server.
  public readonly obsAnchor: [number, number] = [
    PANEL_RIGHT_X,
    -ROBOT_PANEL_H / 2 + 92.7,
  ];
  public readonly actAnchor: [number, number] = [
    PANEL_RIGHT_X,
    -ROBOT_PANEL_H / 2 + 214.7,
  ];

  public constructor(props: RobotProps) {
    super({
      size: [ROBOT_TOTAL_W, ROBOT_TOTAL_H],
      layout: false,
      ...props,
    });

    // White panel with dark border.
    this.add(
      <Rect
        x={PANEL_OFFSET_X}
        size={[ROBOT_PANEL_W, ROBOT_PANEL_H]}
        fill="#FFFFFF"
        stroke={DARK}
        lineWidth={8}
        radius={30}
      />,
    );

    // Robot label box.
    this.add(
      <Rect
        position={robotLocal(PANEL_LOCAL.labelCenter)}
        size={PANEL_LOCAL.labelSize}
        fill={props.labelBg}
        radius={25}
        layout
        alignItems={'center'}
        justifyContent={'center'}
      >
        <Txt
          text={props.label}
          fontFamily={'Helvetica Neue'}
          fontWeight={700}
          fontSize={64}
          fill="#000000"
        />
      </Rect>,
    );

    // Queue panel background (the rounded box around the action chunks).
    this.add(
      <Rect
        position={robotLocal(PANEL_LOCAL.queuePanelCenter)}
        size={PANEL_LOCAL.queuePanelSize}
        fill={null}
        stroke={DARK}
        lineWidth={8}
        radius={30}
      />,
    );

    // "actions left" label inside the queue panel.
    const actionsLabelRightX =
      robotLocal(PANEL_LOCAL.actionsLabelCenter)[0] +
      PANEL_LOCAL.actionsLabelSize[0] / 2 -
      16;
    const actionsLabelCenterY = robotLocal(PANEL_LOCAL.actionsLabelCenter)[1];
    for (const [line, y] of [
      ['actions', actionsLabelCenterY - 23],
      ['left', actionsLabelCenterY + 23],
    ] as const) {
      this.add(
        <Txt
          position={[actionsLabelRightX, y]}
          offset={[1, 0]}
          text={line}
          fontFamily={'Helvetica Neue'}
          fontWeight={700}
          fontSize={40}
          fill="#000000"
        />,
      );
    }

    // The animated action queue chunks.
    this.queue = new ActionQueue({
      position: robotLocal(PANEL_LOCAL.queueChunk0Center),
      color: props.color,
      maxSlots: props.maxSlots,
      initialFill: props.initialFill,
      decay: props.decay,
    });
    this.add(this.queue);

    // Screenshot placeholder to the right. Temporarily disabled to remove the
    // gray box and its layout space between the robot panel and server.
    // this.screenshot = (
    //   <Rect
    //     x={SCREENSHOT_OFFSET_X}
    //     size={[ROBOT_SCREENSHOT_W, ROBOT_SCREENSHOT_H]}
    //     fill={'#E8E8E8'}
    //     stroke={DARK}
    //     lineWidth={10}
    //     radius={25}
    //     clip
    //   >
    //     {props.screenshotChildren}
    //   </Rect>
    // ) as Rect;
    // this.add(this.screenshot);
  }

  public *consumeAction(duration = 0.4) {
    yield* this.queue.consume(duration);
  }

  public *receiveAction(duration = 0.4) {
    yield* this.queue.replenish(duration);
  }

  public *receiveActionToFull(duration = 0.6) {
    yield* this.queue.replenishToFull(duration);
  }
}
