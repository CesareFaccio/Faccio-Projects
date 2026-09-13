// wheel.ts
// The projects carousel: cards ride the rim of a very large, mostly off-screen
// circle, so only the shallow top arc is visible. Rotating the ring carries
// them along that arc, and because each card rotates *with* the ring rather
// than counter-rotating, it tilts as it travels — which is what makes the
// thing read as a wheel rather than a flat row.
//
// The wheel is continuous: slots are laid out all the way round the circle and
// the project list repeats to fill them, so there is no empty space at either
// end and no travel limit. Card spacing is chosen so that fewer cards fit
// across the viewport than there are distinct projects, which is what stops
// the same project ever appearing twice on screen at once.

export interface WheelOptions {
  /** Number of distinct items; the list repeats around the rim. */
  uniqueCount: number;
  /** Builds the card for a given item index. Called once per slot. */
  renderItem(itemIndex: number): HTMLElement;
  /** Fired when a different item reaches the centre. */
  onActiveChange?(itemIndex: number): void;
  /** Below this viewport width the arc is dropped for a flat scroll row. */
  flatBelow?: number;
}

export interface WheelHandle {
  destroy(): void;
}

interface Geometry {
  radius: number;
  stepDeg: number;
  slotsAround: number;
}

function geometryFor(width: number, uniqueCount: number): Geometry {
  // A larger radius means a shallower arc, so the outermost cards drop less
  // far and stay inside the viewport box instead of being clipped mid-card.
  const radius = Math.max(1500, Math.min(2800, width * 1.7));
  const minSpacing = width < 1000 ? 250 : 300;
  // Widen the spacing on large screens if we would otherwise fit more cards
  // across the viewport than we have distinct projects — without this the
  // wheel wraps far enough to show the same project at both edges.
  const spacing = Math.max(minSpacing, width / Math.max(uniqueCount - 0.6, 1));
  const circumference = 2 * Math.PI * radius;
  // Round the slot count to a whole number of full passes through the project
  // list, so the seam where the list repeats lands exactly on a slot boundary.
  const raw = Math.max(uniqueCount, Math.round(circumference / spacing));
  const slotsAround = Math.max(1, Math.round(raw / uniqueCount)) * uniqueCount;
  return { radius, stepDeg: 360 / slotsAround, slotsAround };
}

/** Wraps an angle into (-180, 180]. */
function normalize(deg: number): number {
  let d = deg % 360;
  if (d > 180) d -= 360;
  if (d <= -180) d += 360;
  return d;
}

export function createWheel(viewport: HTMLElement, ring: HTMLElement, options: WheelOptions): WheelHandle {
  const { uniqueCount, renderItem, onActiveChange } = options;
  const flatBelow = options.flatBelow ?? 760;
  if (uniqueCount <= 0) return { destroy() {} };

  let geo = geometryFor(window.innerWidth, uniqueCount);
  let slots: HTMLElement[] = [];
  let angle = 0;
  let velocity = 0;
  let dragging = false;
  let pointerId: number | null = null;
  let lastPointerX = 0;
  let lastMoveTime = 0;
  let dragDistance = 0;
  let snapTarget: number | null = null;
  let activeItem = -1;
  let rafId = 0;
  let wheelSettleTimer = 0;
  let flat = false;

  function buildSlots() {
    ring.textContent = "";
    slots = [];
    // In flat mode only the distinct projects are listed — repeating them in a
    // plain scrolling row would just look like a mistake.
    const count = flat ? uniqueCount : geo.slotsAround;
    for (let i = 0; i < count; i++) {
      const slot = document.createElement("div");
      slot.className = "wheel__slot";
      slot.dataset.item = String(i % uniqueCount);
      slot.appendChild(renderItem(i % uniqueCount));
      ring.appendChild(slot);
      slots.push(slot);
    }
  }

  function applyLayout() {
    const wasFlat = flat;
    flat = window.innerWidth < flatBelow;
    const nextGeo = geometryFor(window.innerWidth, uniqueCount);
    const slotCountChanged = nextGeo.slotsAround !== geo.slotsAround;
    geo = nextGeo;
    viewport.classList.toggle("is-flat", flat);

    if (slots.length === 0 || flat !== wasFlat || (!flat && slotCountChanged)) {
      buildSlots();
    }

    if (flat) {
      ring.style.transform = "";
      ring.style.height = "";
      for (const slot of slots) {
        slot.style.transform = "";
        slot.style.transformOrigin = "";
        slot.style.opacity = "";
        slot.style.zIndex = "";
      }
      return;
    }

    // The ring's box is exactly one radius tall, so its bottom edge sits at the
    // circle's centre and rotating it sweeps every slot around that point.
    ring.style.height = `${geo.radius}px`;
    for (let i = 0; i < slots.length; i++) {
      slots[i].style.transformOrigin = `50% ${geo.radius}px`;
      slots[i].style.transform = `translateX(-50%) rotate(${i * geo.stepDeg}deg)`;
    }
    render();
  }

  function render() {
    if (flat) return;
    ring.style.transform = `rotate(${-angle}deg)`;
    let nearestSlot = 0;
    let nearestDelta = Infinity;
    for (let i = 0; i < slots.length; i++) {
      const delta = normalize(i * geo.stepDeg - angle);
      const offset = Math.abs(delta) / geo.stepDeg;
      if (Math.abs(delta) < Math.abs(nearestDelta)) {
        nearestDelta = delta;
        nearestSlot = i;
      }
      // Cards more than a few steps out are round the rim and mostly decorative.
      const dim = Math.min(offset / 3.2, 1);
      slots[i].style.opacity = String(1 - dim * 0.78);
      slots[i].style.zIndex = String(100 - Math.round(offset * 10));
      slots[i].classList.toggle("is-active", false);
    }
    slots[nearestSlot]?.classList.add("is-active");
    const item = nearestSlot % uniqueCount;
    if (item !== activeItem) {
      activeItem = item;
      onActiveChange?.(item);
    }
  }

  function tick() {
    rafId = requestAnimationFrame(tick);
    if (dragging || flat) return;

    if (snapTarget !== null) {
      const diff = snapTarget - angle;
      if (Math.abs(diff) < 0.02) {
        angle = snapTarget;
        snapTarget = null;
      } else {
        angle += diff * 0.16;
      }
      render();
      return;
    }

    if (Math.abs(velocity) > 0.01) {
      angle += velocity;
      velocity *= 0.93;
      render();
      if (Math.abs(velocity) <= 0.01) beginSnap();
    }
  }

  function beginSnap() {
    velocity = 0;
    snapTarget = Math.round(angle / geo.stepDeg) * geo.stepDeg;
  }

  /** Rotates to the nearest slot showing `itemIndex`, going whichever way is shorter. */
  function goToItem(itemIndex: number) {
    const currentSlot = Math.round(angle / geo.stepDeg);
    const currentItem = ((currentSlot % uniqueCount) + uniqueCount) % uniqueCount;
    let stepsForward = (itemIndex - currentItem + uniqueCount) % uniqueCount;
    if (stepsForward > uniqueCount / 2) stepsForward -= uniqueCount;
    velocity = 0;
    snapTarget = (currentSlot + stepsForward) * geo.stepDeg;
  }

  function goBySlots(delta: number) {
    velocity = 0;
    snapTarget = (Math.round(angle / geo.stepDeg) + delta) * geo.stepDeg;
  }

  // ── Dragging ─────────────────────────────────────────────────────────────
  function onPointerDown(e: PointerEvent) {
    if (flat || e.button !== 0) return;
    dragging = true;
    pointerId = e.pointerId;
    lastPointerX = e.clientX;
    lastMoveTime = performance.now();
    dragDistance = 0;
    velocity = 0;
    snapTarget = null;
    viewport.classList.add("is-dragging");
    viewport.setPointerCapture(e.pointerId);
  }

  function onPointerMove(e: PointerEvent) {
    if (!dragging || e.pointerId !== pointerId) return;
    const dx = e.clientX - lastPointerX;
    lastPointerX = e.clientX;
    dragDistance += Math.abs(dx);
    const now = performance.now();
    const dt = Math.max(now - lastMoveTime, 1);
    lastMoveTime = now;
    // Dragging left should advance the wheel, hence the inverted sign.
    const deltaDeg = (-dx / geo.radius) * (180 / Math.PI) * 2.4;
    angle += deltaDeg;
    velocity = (deltaDeg / dt) * 16;
    render();
  }

  function endDrag(e: PointerEvent) {
    if (!dragging || e.pointerId !== pointerId) return;
    dragging = false;
    pointerId = null;
    viewport.classList.remove("is-dragging");
    if (viewport.hasPointerCapture(e.pointerId)) viewport.releasePointerCapture(e.pointerId);
    if (Math.abs(velocity) < 0.12) beginSnap();
  }

  // ── Keyboard ─────────────────────────────────────────────────────────────
  function onKeyDown(e: KeyboardEvent) {
    if (flat) return;
    if (e.key === "ArrowRight") {
      e.preventDefault();
      goBySlots(1);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      goBySlots(-1);
    }
  }

  // ── Trackpad ─────────────────────────────────────────────────────────────
  // Only clearly-horizontal gestures are claimed. Vertical scrolling belongs to
  // the page; hijacking it is the most irritating thing a carousel can do.
  function onWheelEvent(e: WheelEvent) {
    if (flat) return;
    if (Math.abs(e.deltaX) <= Math.abs(e.deltaY)) return;
    e.preventDefault();
    snapTarget = null;
    angle += (e.deltaX / geo.radius) * (180 / Math.PI) * 2.4;
    render();
    window.clearTimeout(wheelSettleTimer);
    wheelSettleTimer = window.setTimeout(beginSnap, 140);
  }

  // A click on an off-centre card brings it to the centre rather than opening
  // it; once it is centred the link behaves normally. A click that ends a drag
  // is swallowed, so flicking the wheel never navigates by accident.
  function onClickCapture(e: MouseEvent) {
    if (flat) return;
    const slot = (e.target as HTMLElement).closest(".wheel__slot") as HTMLElement | null;
    if (!slot) return;
    if (dragDistance > 6) {
      e.preventDefault();
      e.stopPropagation();
      dragDistance = 0;
      return;
    }
    if (!slot.classList.contains("is-active")) {
      e.preventDefault();
      e.stopPropagation();
      const index = slots.indexOf(slot);
      if (index !== -1) {
        velocity = 0;
        snapTarget = Math.round((angle + normalize(index * geo.stepDeg - angle)) / geo.stepDeg) * geo.stepDeg;
      }
    }
  }

  function onResize() {
    applyLayout();
  }

  viewport.addEventListener("pointerdown", onPointerDown);
  viewport.addEventListener("pointermove", onPointerMove);
  viewport.addEventListener("pointerup", endDrag);
  viewport.addEventListener("pointercancel", endDrag);
  viewport.addEventListener("keydown", onKeyDown);
  viewport.addEventListener("wheel", onWheelEvent, { passive: false });
  viewport.addEventListener("click", onClickCapture, true);
  window.addEventListener("resize", onResize);

  applyLayout();
  rafId = requestAnimationFrame(tick);

  return {
    destroy() {
      cancelAnimationFrame(rafId);
      window.clearTimeout(wheelSettleTimer);
      viewport.removeEventListener("pointerdown", onPointerDown);
      viewport.removeEventListener("pointermove", onPointerMove);
      viewport.removeEventListener("pointerup", endDrag);
      viewport.removeEventListener("pointercancel", endDrag);
      viewport.removeEventListener("keydown", onKeyDown);
      viewport.removeEventListener("wheel", onWheelEvent);
      viewport.removeEventListener("click", onClickCapture, true);
      window.removeEventListener("resize", onResize);
    },
  };
}
export type { Geometry };
