interface MathVisualProps {
  query: string;
  answer: string;
}

function numberPart(value: string, fallback: number): number {
  if (value === "" || value === "+") return fallback;
  if (value === "-") return -fallback;
  return Number(value);
}

export function MathVisual({ query, answer }: MathVisualProps) {
  const content = `${query}\n${answer}`.replace(/−/g, "-");
  const linear = content.match(
    /y\s*=\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+)?)\s*x\s*([+-]\s*\d+(?:\.\d+)?)?/i,
  );
  if (linear) {
    const slope = numberPart(linear[1], 1);
    const intercept = Number((linear[2] || "0").replace(/\s+/g, ""));
    if (Number.isFinite(slope) && Number.isFinite(intercept)) {
      const mapX = (x: number) => 32 + ((x + 5) / 10) * 456;
      const mapY = (y: number) => 18 + ((6 - y) / 12) * 224;
      const points = Array.from({ length: 41 }, (_, index) => {
        const x = -5 + index / 4;
        return `${mapX(x)},${mapY(slope * x + intercept)}`;
      }).join(" ");
      return (
        <figure className="math-visual" aria-label={`函数 y = ${slope}x + ${intercept} 的图像`}>
          <figcaption>函数图像</figcaption>
          <svg viewBox="0 0 520 260" role="img">
            <title>{`y = ${slope}x + ${intercept}`}</title>
            <defs>
              <clipPath id="plot-clip">
                <rect x="32" y="18" width="456" height="224" />
              </clipPath>
            </defs>
            <line className="grid-axis" x1="32" y1={mapY(0)} x2="488" y2={mapY(0)} />
            <line className="grid-axis" x1={mapX(0)} y1="18" x2={mapX(0)} y2="242" />
            <polyline className="function-line" points={points} clipPath="url(#plot-clip)" />
            <circle className="function-point" cx={mapX(0)} cy={mapY(intercept)} r="4" />
            <text x={mapX(0) + 8} y={mapY(intercept) - 8}>{`(0, ${intercept})`}</text>
            <text className="axis-label" x="492" y={mapY(0) - 6}>x</text>
            <text className="axis-label" x={mapX(0) + 7} y="16">y</text>
          </svg>
        </figure>
      );
    }
  }

  const triangle = content.match(/△\s*([A-Z])([A-Z])([A-Z])/);
  if (triangle) {
    const [, a, b, c] = triangle;
    return (
      <figure className="math-visual triangle-visual" aria-label={`三角形 ${a}${b}${c} 示意图`}>
        <figcaption>题目示意图 <small>不按比例</small></figcaption>
        <svg viewBox="0 0 520 250" role="img">
          <title>{`三角形 ${a}${b}${c}`}</title>
          <polygon points="260,24 70,220 455,220" />
          <circle cx="260" cy="24" r="4" />
          <circle cx="70" cy="220" r="4" />
          <circle cx="455" cy="220" r="4" />
          <text x="260" y="17" textAnchor="middle">{a}</text>
          <text x="54" y="238">{b}</text>
          <text x="462" y="238">{c}</text>
        </svg>
      </figure>
    );
  }
  return null;
}
