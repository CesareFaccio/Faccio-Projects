interface Project {
  name: string;
  description: string;
  href: string;
  status: "live" | "in progress";
}

// Add new projects here as they're built — this is the only place the
// landing page needs to be touched to add a new card.
const projects: Project[] = [
  {
    name: "BetaScope",
    description:
      "Upload a climbing video and see pose tracking, hold detection, and biomechanical force analysis — all computed in your browser.",
    href: "/betascope/",
    status: "in progress",
  },
];

const gridEl = document.getElementById("project-grid")!;

for (const p of projects) {
  const card = document.createElement("a");
  card.className = "card";
  card.href = p.href;
  card.innerHTML = `
    <h2>${p.name}</h2>
    <p>${p.description}</p>
    <span class="badge">${p.status}</span>
  `;
  gridEl.appendChild(card);
}
