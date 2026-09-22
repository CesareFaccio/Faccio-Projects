import { startFluid } from "./fluid";
import { createWheel } from "./wheel";
import { makeWindowsDraggable } from "./windows";

interface Project {
  name: string;
  /** Short line shown on the card itself. */
  blurb: string;
  /** Longer line shown under the wheel for whichever card is centred. */
  detail: string;
  href: string | null;
  status: "live" | "in progress" | "planned";
  /** Two or three words set in the card's large display type. */
  mark: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// The project list. This is the only place the landing page needs editing to
// add, rename or reorder a project.
//
// The entries below marked "planned" are PLACEHOLDERS — rename them, rewrite
// their text, and set an href once each one exists. Delete any you don't want;
// the wheel lays itself out around however many entries are here.
// ─────────────────────────────────────────────────────────────────────────────
const projects: Project[] = [
  {
    name: "Crux Vision",
    mark: "Crux\nVision",
    blurb: "Climbing video analysis",
    detail:
      "Upload a climbing video and see pose tracking, hold detection and biomechanical force analysis — computed entirely in your browser, with nothing uploaded anywhere.",
    href: "crux-vision/",
    status: "in progress",
  },
  {
    name: "Project Two",
    mark: "Two",
    blurb: "Placeholder",
    detail: "Placeholder card — replace this entry in src/main.ts with your next project.",
    href: null,
    status: "planned",
  },
  {
    name: "Project Three",
    mark: "Three",
    blurb: "Placeholder",
    detail: "Placeholder card — replace this entry in src/main.ts with your next project.",
    href: null,
    status: "planned",
  },
  {
    name: "Project Four",
    mark: "Four",
    blurb: "Placeholder",
    detail: "Placeholder card — replace this entry in src/main.ts with your next project.",
    href: null,
    status: "planned",
  },
  {
    name: "Project Five",
    mark: "Five",
    blurb: "Placeholder",
    detail: "Placeholder card — replace this entry in src/main.ts with your next project.",
    href: null,
    status: "planned",
  },
];

// ── Hero ─────────────────────────────────────────────────────────────────────
const heroCanvas = document.getElementById("fluid-canvas") as HTMLCanvasElement | null;
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// Hiding the canvas uncovers the message underneath it, so the two reasons the
// simulation might not run need to say which one it was — "no WebGL2" is a lie
// when the visitor simply asked for less motion.
const heroFallback = document.querySelector<HTMLElement>(".win__fallback");

if (heroCanvas && !prefersReducedMotion) {
  // startFluid() returns null when WebGL2 or float render targets are missing.
  const handle = startFluid(heroCanvas);
  if (!handle) heroCanvas.classList.add("is-unsupported");
} else if (heroCanvas) {
  heroCanvas.classList.add("is-unsupported");
  if (heroFallback) {
    heroFallback.textContent = "Simulation paused \u2014 your system is set to reduce motion.";
  }
}

// ── Project wheel ────────────────────────────────────────────────────────────
const ring = document.getElementById("wheel-ring") as HTMLElement;
const viewport = document.getElementById("wheel-viewport") as HTMLElement;
const meta = document.getElementById("wheel-meta") as HTMLElement;

/** Builds one card. Everything sits inside the card's own box: a label hung
 *  underneath would rotate with the card and collide with its neighbours. */
function buildCard(itemIndex: number): HTMLElement {
  const project = projects[itemIndex];

  // A card with somewhere to go is a link; a placeholder is inert, since
  // marking it up as a link nobody can follow is just a broken promise.
  const card = document.createElement(project.href ? "a" : "div");
  card.className = `card card--${project.status.replace(/\s+/g, "-")}`;
  if (project.href && card instanceof HTMLAnchorElement) card.href = project.href;

  const markLines = project.mark
    .split("\n")
    .map((line) => `<span>${line}</span>`)
    .join("");

  // Each card is a miniature window: title bar, body, status strip.
  card.innerHTML = `
    <div class="card__bar"><span class="card__bar-title">${project.name}</span></div>
    <div class="card__in">
      <div class="card__mark">${markLines}</div>
      <div class="card__foot">
        <span class="card__blurb">${project.blurb}</span>
        ${
          project.href
            ? '<span class="card__cta">Open &#9654;</span>'
            : '<span class="card__cta card__cta--muted">Coming soon</span>'
        }
      </div>
    </div>
    <div class="card__status">${project.status}</div>
  `;
  return card;
}

function showMeta(itemIndex: number) {
  const project = projects[itemIndex];
  if (project) meta.textContent = project.detail;
}

createWheel(viewport, ring, {
  uniqueCount: projects.length,
  renderItem: buildCard,
  onActiveChange: showMeta,
});
showMeta(0);

// ── Desktop ──────────────────────────────────────────────────────────────────
makeWindowsDraggable(document);

// ── Footer ───────────────────────────────────────────────────────────────────
const yearEl = document.getElementById("year");
if (yearEl) yearEl.textContent = String(new Date().getFullYear());
