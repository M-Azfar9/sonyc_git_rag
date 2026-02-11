
"use client";

import { useEffect, useMemo, useState } from "react";
import Particles, { initParticlesEngine } from "@tsparticles/react";
import type { Container, IOptions } from "@tsparticles/engine";
import { loadSlim } from "@tsparticles/slim";

export const MotionBackground = () => {
  const [init, setInit] = useState(false);

  useEffect(() => {
    initParticlesEngine(async (engine) => {
      await loadSlim(engine);
    }).then(() => {
      setInit(true);
    });
  }, []);

  const particlesLoaded = async (container?: Container): Promise<void> => {
    console.log(container);
  };

  const options = useMemo(
    () => ({
      background: {
        color: {
          value: "transparent",
        },
      },
      fpsLimit: 60,
      interactivity: {
        events: {
          onHover: {
            enable: false,
          },
        },
      },
      particles: {
        color: {
          value: "hsl(var(--primary))",
        },
        links: {
          enable: false,
        },
        move: {
          direction: "top",
          enable: true,
          outModes: {
            default: "out",
          },
          random: true,
          speed: 0.5,
          straight: false,
        },
        number: {
          density: {
            enable: true,
            area: 800,
          },
          value: 20,
        },
        opacity: {
          value: { min: 0.1, max: 0.4 },
          animation: {
            enable: true,
            speed: 1,
            minimumValue: 0.05,
          },
        },
        shape: {
          type: "circle",
        },
        size: {
          value: { min: 20, max: 150 },
          animation: {
            enable: true,
            speed: 5,
            minimumValue: 10,
          },
        },
      },
      detectRetina: true,
    } as unknown as IOptions),
    [],
  );

  if (init) {
    return (
      <Particles
        id="tsparticles"
        particlesLoaded={particlesLoaded}
        options={options}
        className="fixed inset-0 -z-10"
      />
    );
  }

  return null;
};
