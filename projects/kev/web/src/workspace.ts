type View = "model" | "data" | "evaluation" | "capabilities";
type Direction = "left" | "right" | "up" | "down";

const locations: Record<View, readonly [number, number]> = {
  model: [0, 0],
  data: [-1, 0],
  evaluation: [1, 0],
  capabilities: [0, -1],
};
const routes: Record<View, Partial<Record<Direction, View>>> = {
  model: { left: "data", right: "evaluation", up: "capabilities" },
  data: { right: "model" },
  evaluation: { left: "model" },
  capabilities: { down: "model" },
};
const labels: Record<View, string> = {
  model: "Model",
  data: "Data",
  evaluation: "Evaluation",
  capabilities: "Capabilities",
};

function isDirection(value: string | undefined): value is Direction {
  return (
    value === "left" || value === "right" || value === "up" || value === "down"
  );
}

export function initializeWorkspace(): void {
  const plane = document.getElementById("workspace-plane");
  if (!(plane instanceof HTMLElement)) return;
  const workspacePlane = plane;
  const buttons = document.querySelectorAll<HTMLButtonElement>(
    "#workspace-navigation button[data-direction]",
  );
  let current: View = "model";

  function navigate(next: View, focus = true): void {
    if (next !== current) {
      const play = document.getElementById("play");
      if (play instanceof HTMLButtonElement && play.textContent === "Stop")
        play.click();
      const inspector = document.getElementById("inspector");
      if (inspector instanceof HTMLDialogElement) inspector.close();
    }
    current = next;
    const [x, y] = locations[next];
    workspacePlane.style.transform = `translate(${-x * 100}%, ${-y * 100}%)`;
    for (const view of Object.keys(locations)) {
      const panel = document.getElementById(`view-${view}`);
      if (!(panel instanceof HTMLElement)) continue;
      panel.inert = view !== next;
      panel.setAttribute("aria-hidden", String(view !== next));
      if (view === next && focus) panel.focus({ preventScroll: true });
    }
    for (const button of buttons) {
      const direction = button.dataset.direction;
      if (!isDirection(direction)) continue;
      const target = routes[next][direction];
      button.hidden = target === undefined;
      button.classList.remove("near-edge");
      if (!target) continue;
      button.setAttribute("aria-label", `Go to ${labels[target]}`);
      button.setAttribute("aria-controls", `view-${target}`);
      const label = button.querySelector(".edge-label");
      if (label) label.textContent = labels[target];
    }
  }

  for (const button of buttons) {
    button.addEventListener("click", () => {
      const direction = button.dataset.direction;
      if (!isDirection(direction)) return;
      const target = routes[current][direction];
      if (target) navigate(target);
    });
  }

  document.addEventListener("pointermove", (event) => {
    const near: Record<Direction, boolean> = {
      left: event.clientX < 120,
      right: event.clientX > window.innerWidth - 120,
      up: event.clientY < 100,
      down: event.clientY > window.innerHeight - 100,
    };
    for (const button of buttons) {
      const direction = button.dataset.direction;
      if (isDirection(direction))
        button.classList.toggle("near-edge", near[direction]);
    }
  });
  document.documentElement.addEventListener("pointerleave", () => {
    for (const button of buttons) button.classList.remove("near-edge");
  });
  navigate("model", false);
}

initializeWorkspace();
