import React, { useEffect, useRef } from "react";

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  radius: number;
  baseAlpha: number;
  alpha: number;
  pulseSpeed: number;
  pulsePhase: number;
  depth: number; // 0.5 to 1.5 for parallax
}

interface TechNode {
  x: number;
  y: number;
  size: number;
  rotation: number;
  rotationSpeed: number;
  alpha: number;
  shape: "hex" | "cross" | "box";
}

export const GeospatialParticleBackground: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const mouse = { x: width / 2, y: height / 2, targetX: width / 2, targetY: height / 2 };

    // Create Particles
    const count = Math.min(65, Math.floor((width * height) / 22000));
    const particles: Particle[] = [];

    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random() - 0.5) * 0.25,
        vy: (Math.random() - 0.5) * 0.25,
        radius: Math.random() * 1.6 + 0.8,
        baseAlpha: Math.random() * 0.35 + 0.15,
        alpha: 0.2,
        pulseSpeed: Math.random() * 0.02 + 0.005,
        pulsePhase: Math.random() * Math.PI * 2,
        depth: Math.random() * 0.8 + 0.6,
      });
    }

    // Floating abstract tech geometry elements
    const techNodes: TechNode[] = [
      { x: width * 0.15, y: height * 0.25, size: 28, rotation: 0, rotationSpeed: 0.002, alpha: 0.12, shape: "hex" },
      { x: width * 0.85, y: height * 0.3, size: 22, rotation: 0.5, rotationSpeed: -0.0015, alpha: 0.10, shape: "cross" },
      { x: width * 0.75, y: height * 0.75, size: 34, rotation: 1.2, rotationSpeed: 0.0018, alpha: 0.09, shape: "hex" },
      { x: width * 0.2, y: height * 0.8, size: 20, rotation: 0.8, rotationSpeed: -0.0025, alpha: 0.11, shape: "box" },
    ];

    const handleMouseMove = (e: MouseEvent) => {
      mouse.targetX = e.clientX;
      mouse.targetY = e.clientY;
    };

    const handleResize = () => {
      if (!canvas) return;
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    };

    window.addEventListener("mousemove", handleMouseMove, { passive: true });
    window.addEventListener("resize", handleResize);

    const drawHexagon = (c: CanvasRenderingContext2D, x: number, y: number, r: number) => {
      c.beginPath();
      for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 3) * i;
        const hx = x + r * Math.cos(angle);
        const hy = y + r * Math.sin(angle);
        if (i === 0) c.moveTo(hx, hy);
        else c.lineTo(hx, hy);
      }
      c.closePath();
    };

    const render = () => {
      // Smooth mouse follow for gentle parallax
      mouse.x += (mouse.targetX - mouse.x) * 0.03;
      mouse.y += (mouse.targetY - mouse.y) * 0.03;

      const offsetX = (mouse.x - width / 2) * 0.02;
      const offsetY = (mouse.y - height / 2) * 0.02;

      // Dark black to charcoal gradient background
      const grad = ctx.createRadialGradient(
        width / 2, height / 2, 50,
        width / 2, height / 2, Math.max(width, height) * 0.8
      );
      grad.addColorStop(0, "#0c1017");
      grad.addColorStop(0.6, "#070a0f");
      grad.addColorStop(1, "#040609");

      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, width, height);

      // 1. Draw subtle background coordinate grid
      ctx.strokeStyle = "rgba(51, 65, 85, 0.07)";
      ctx.lineWidth = 1;
      const gridSize = 80;
      const startX = (offsetX * 0.5) % gridSize;
      const startY = (offsetY * 0.5) % gridSize;

      ctx.beginPath();
      for (let x = startX; x < width; x += gridSize) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }
      for (let y = startY; y < height; y += gridSize) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }
      ctx.stroke();

      // 2. Draw floating abstract tech shapes
      techNodes.forEach((node) => {
        node.rotation += node.rotationSpeed;
        const px = node.x - offsetX * 0.8;
        const py = node.y - offsetY * 0.8;

        ctx.save();
        ctx.translate(px, py);
        ctx.rotate(node.rotation);
        ctx.strokeStyle = `rgba(100, 116, 139, ${node.alpha})`;
        ctx.lineWidth = 1;

        if (node.shape === "hex") {
          drawHexagon(ctx, 0, 0, node.size);
          ctx.stroke();
          // Inner dot
          ctx.fillStyle = `rgba(0, 229, 255, ${node.alpha * 1.5})`;
          ctx.beginPath();
          ctx.arc(0, 0, 1.5, 0, Math.PI * 2);
          ctx.fill();
        } else if (node.shape === "cross") {
          const s = node.size * 0.6;
          ctx.beginPath();
          ctx.moveTo(-s, 0);
          ctx.lineTo(s, 0);
          ctx.moveTo(0, -s);
          ctx.lineTo(0, s);
          ctx.stroke();
        } else if (node.shape === "box") {
          const s = node.size * 0.7;
          ctx.strokeRect(-s / 2, -s / 2, s, s);
        }
        ctx.restore();
      });

      // 3. Update & Draw Particles and Connecting Lines
      const maxDistance = 120;

      // Update positions
      particles.forEach((p) => {
        p.pulsePhase += p.pulseSpeed;
        p.alpha = p.baseAlpha + Math.sin(p.pulsePhase) * 0.12;

        p.x += p.vx;
        p.y += p.vy;

        // Wrap around borders
        if (p.x < -10) p.x = width + 10;
        if (p.x > width + 10) p.x = -10;
        if (p.y < -10) p.y = height + 10;
        if (p.y > height + 10) p.y = -10;
      });

      // Draw connecting lines between close particles
      ctx.lineWidth = 0.75;
      for (let i = 0; i < particles.length; i++) {
        const p1 = particles[i];
        const p1x = p1.x - offsetX * p1.depth;
        const p1y = p1.y - offsetY * p1.depth;

        for (let j = i + 1; j < particles.length; j++) {
          const p2 = particles[j];
          const p2x = p2.x - offsetX * p2.depth;
          const p2y = p2.y - offsetY * p2.depth;

          const dx = p1x - p2x;
          const dy = p1y - p2y;
          const dist = Math.sqrt(dx * dx + dy * dy);

          if (dist < maxDistance) {
            const lineAlpha = (1 - dist / maxDistance) * 0.18 * Math.min(p1.alpha, p2.alpha);
            ctx.strokeStyle = `rgba(148, 163, 184, ${lineAlpha})`;
            ctx.beginPath();
            ctx.moveTo(p1x, p1y);
            ctx.lineTo(p2x, p2y);
            ctx.stroke();
          }
        }

        // Draw particle dot
        ctx.fillStyle = `rgba(148, 163, 184, ${p1.alpha})`;
        ctx.beginPath();
        ctx.arc(p1x, p1y, p1.radius, 0, Math.PI * 2);
        ctx.fill();

        // Subtle cyan core on high-depth particles
        if (p1.depth > 1.1) {
          ctx.fillStyle = `rgba(0, 229, 255, ${p1.alpha * 0.6})`;
          ctx.beginPath();
          ctx.arc(p1x, p1y, p1.radius * 0.6, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      animId = requestAnimationFrame(render);
    };

    render();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="fixed inset-0 pointer-events-none z-0 w-full h-full"
      aria-hidden="true"
    />
  );
};
