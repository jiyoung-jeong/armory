export const BLUE = '#37A3D2';
export const GREY = '#777B7F';
export const DARK = '#3A3A3A';
export const RED = '#F94144';
export const YELLOW = '#F9C74F';
export const BG = '#FFFFFF';

export function withAlpha(color: string, alpha: number): string {
  const r = parseInt(color.slice(1, 3), 16);
  const g = parseInt(color.slice(3, 5), 16);
  const b = parseInt(color.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
