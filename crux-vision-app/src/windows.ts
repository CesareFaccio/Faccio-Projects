// windows.ts
// Lets the desktop's windows be picked up by their title bars and moved, the
// way the real thing did.
//
// A copy of ../../src/windows.ts. The two apps are separate Vite builds with
// separate roots, so they duplicate their shared front-end the same way their
// stylesheets do — a change here wants the same change there.
//
// Two rules keep this from fighting the rest of the page:
//
//  1. It moves windows with `transform`, never by changing their layout
//     position. The page is still an ordinary document flow — the projects
//     window is what the "Projects" link scrolls to, and the hero grid still
//     decides how big everything is. Dragging only paints it somewhere else.
//
//  2. Only the title bar starts a drag, and only with a fine pointer on a wide
//     screen. On a phone the windows are stacked full-width with nowhere to go,
//     and a drag handle there would just be a way to lose the page.

const MIN_WIDTH = 900;

interface Dragging {
  el: HTMLElement;
  pointerId: number;
  startX: number;
  startY: number;
  baseX: number;
  baseY: number;
  /** Where the window sits with no offset applied, measured once at grab time
   *  so the clamp below never has to reason about its own output. */
  homeLeft: number;
  homeTop: number;
  width: number;
}

export function makeWindowsDraggable(root: ParentNode = document): () => void {
  const windows = Array.from(root.querySelectorAll<HTMLElement>("[data-window]"));
  if (windows.length === 0) return () => {};

  const finePointer = window.matchMedia("(hover: hover) and (pointer: fine)");
  const offsets = new WeakMap<HTMLElement, { x: number; y: number }>();
  let current: Dragging | null = null;
  let front = 20;

  function enabled(): boolean {
    return finePointer.matches && window.innerWidth >= MIN_WIDTH;
  }

  function offsetOf(el: HTMLElement) {
    return offsets.get(el) ?? { x: 0, y: 0 };
  }

  function place(el: HTMLElement, x: number, y: number) {
    offsets.set(el, { x, y });
    el.style.transform = x === 0 && y === 0 ? "" : `translate(${x}px, ${y}px)`;
  }

  function onPointerDown(e: PointerEvent) {
    if (!enabled() || e.button !== 0 || current) return;
    const handle = (e.target as HTMLElement).closest<HTMLElement>("[data-drag-handle]");
    if (!handle) return;
    const el = handle.closest<HTMLElement>("[data-window]");
    if (!el) return;

    // The close and zoom boxes are decoration, but grabbing the window by one
    // of them still feels wrong, so they are not drag starts either.
    if ((e.target as HTMLElement).closest(".win__close, .win__zoom")) return;

    const base = offsetOf(el);
    const rect = el.getBoundingClientRect();
    current = {
      el,
      pointerId: e.pointerId,
      startX: e.clientX,
      startY: e.clientY,
      baseX: base.x,
      baseY: base.y,
      homeLeft: rect.left - base.x,
      homeTop: rect.top - base.y,
      width: rect.width,
    };
    for (const w of windows) w.classList.remove("is-front");
    el.classList.add("is-front");
    el.style.zIndex = String(++front);
    handle.setPointerCapture(e.pointerId);
    e.preventDefault();
  }

  function onPointerMove(e: PointerEvent) {
    if (!current || e.pointerId !== current.pointerId) return;
    const { el, startX, startY, baseX, baseY, homeLeft, homeTop, width } = current;

    let x = baseX + (e.clientX - startX);
    const y = baseY + (e.clientY - startY);

    // Keep the title bar reachable: a window can go mostly off-screen, but
    // never so far that there is nothing left to grab it by.
    const minVisible = 90;
    x = Math.min(x, window.innerWidth - minVisible - homeLeft);
    x = Math.max(x, minVisible - width - homeLeft);
    // Vertically, never up behind the menu bar. There is no lower bound: the
    // page scrolls, so dragging a window down is harmless.
    place(el, x, Math.max(y, 26 - homeTop));
  }

  function endDrag(e: PointerEvent) {
    if (!current || e.pointerId !== current.pointerId) return;
    const handle = current.el.querySelector<HTMLElement>("[data-drag-handle]");
    if (handle?.hasPointerCapture(e.pointerId)) handle.releasePointerCapture(e.pointerId);
    current = null;
  }

  // Narrowing the viewport puts the windows back where the layout wants them;
  // an offset saved at desktop width would otherwise push a stacked phone
  // layout off the side of the screen.
  function onResize() {
    if (enabled()) return;
    for (const el of windows) place(el, 0, 0);
  }

  for (const el of windows) {
    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", endDrag);
    el.addEventListener("pointercancel", endDrag);
  }
  window.addEventListener("resize", onResize);

  return () => {
    for (const el of windows) {
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", endDrag);
      el.removeEventListener("pointercancel", endDrag);
      place(el, 0, 0);
      el.classList.remove("is-front");
      el.style.zIndex = "";
    }
    window.removeEventListener("resize", onResize);
  };
}
