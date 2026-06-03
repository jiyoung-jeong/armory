import {Layout, LayoutProps, Rect} from '@motion-canvas/2d';
import {DARK} from '../colors';

export interface ServerProps extends LayoutProps {}

export const SERVER_W = 1699;
export const SERVER_H = 1512;

export class Server extends Layout {
  public constructor(props?: ServerProps) {
    super({size: [SERVER_W, SERVER_H], layout: false, ...props});

    this.add(
      <Rect
        size={[SERVER_W, SERVER_H]}
        fill="#FFFFFF"
        stroke={DARK}
        lineWidth={8}
        radius={25}
      />,
    );
  }
}
