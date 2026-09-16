import { cn } from "@/lib/utils";

interface CircularLoaderProps {
  size?: number;
  strokeWidth?: number;
  className?: string;
  label?: string;
  labelClassName?: string;
  color?: string;
  trackColor?: string;
}

/**
 * Circular arc loader — a blue spinning arc over a light gray track.
 * Matches the design reference: rounded stroke ends, ~70% arc, smooth spin.
 */
export function CircularLoader({
  size = 40,
  strokeWidth = 4,
  className,
  color = "#1565EF",
  trackColor = "#E4E7EC",
}: CircularLoaderProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  // ~70% arc visible
  const dash = circumference * 0.7;
  const gap = circumference - dash;

  return (
    <svg
      className={cn("animate-spin", className)}
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      role="status"
      aria-label="Loading"
    >
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke={trackColor}
        strokeWidth={strokeWidth}
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={radius}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeDasharray={`${dash} ${gap}`}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
    </svg>
  );
}

export function CircularLoaderWithLabel({
  label = "Loading...",
  labelClassName,
  ...props
}: CircularLoaderProps) {
  return (
    <div className="flex flex-col items-center justify-center gap-3">
      <CircularLoader {...props} />
      <span className={cn("text-sm text-muted-foreground", labelClassName)}>
        {label}
      </span>
    </div>
  );
}
