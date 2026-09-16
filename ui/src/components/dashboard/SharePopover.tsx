import { X, UsersPlus, User01, ChevronDown, Link01, Share07 } from "@untitledui/icons";

type Member = { name: string; role: string; initials: string };

const MEMBERS: Member[] = [
  { name: "Marcus Rivera", role: "Software Engineer", initials: "MR" },
  { name: "Derek Kim", role: "Data Analyst", initials: "DK" },
  { name: "Lila Thompson", role: "Marketing Specialist", initials: "LT" },
  { name: "Jasmine Patel", role: "Product Manager", initials: "JP" },
];

export function SharePopover({ onClose }: { onClose: () => void }) {
  return (
    <div
      role="dialog"
      aria-label="Share with people"
      className="absolute right-0 top-12 z-30 w-[360px] rounded-2xl border border-secondary bg-primary p-5 shadow-lg"
    >
      <div className="flex items-start justify-between">
        <span className="grid size-10 place-items-center rounded-lg border border-secondary bg-primary shadow-xs">
          <UsersPlus className="size-5 text-secondary" />
        </span>
        <button onClick={onClose} aria-label="Close" className="rounded p-1 text-fg-quaternary hover:bg-primary_hover">
          <X className="size-5" />
        </button>
      </div>

      <h3 className="mt-3 text-md font-semibold text-primary">Share with people</h3>
      <p className="mt-0.5 text-sm text-tertiary">Manage who can view or collaborate in the analysis</p>

      <div className="mt-4">
        <label className="text-sm font-medium text-secondary">Team member</label>
        <button
          type="button"
          className="mt-1.5 flex w-full items-center justify-between rounded-lg border border-secondary bg-primary px-3 py-2 text-sm text-tertiary shadow-xs hover:bg-primary_hover"
        >
          <span className="flex items-center gap-2">
            <User01 className="size-4" />
            Select team member
          </span>
          <ChevronDown className="size-4" />
        </button>
      </div>

      <div className="mt-5">
        <p className="text-sm font-semibold text-primary">Members With Access</p>
        <ul className="mt-3 flex flex-col gap-3">
          {MEMBERS.map((m) => (
            <li key={m.name} className="flex items-center gap-3">
              <span className="grid size-9 shrink-0 place-items-center rounded-full bg-[#1565EF]/10 text-xs font-semibold text-[#1565EF]">
                {m.initials}
              </span>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-semibold text-primary">{m.name}</p>
                <p className="truncate text-xs text-tertiary">{m.role}</p>
              </div>
              <button className="flex items-center gap-1 text-sm font-medium text-secondary hover:text-primary">
                View <ChevronDown className="size-3.5" />
              </button>
            </li>
          ))}
        </ul>
      </div>

      <div className="mt-5 flex items-center gap-2">
        <button className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-secondary bg-primary px-3 py-2 text-sm font-semibold text-secondary shadow-xs hover:bg-primary_hover">
          <Link01 className="size-4" />
          Copy Link
        </button>
        <button className="inline-flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-[#1565EF] px-3 py-2 text-sm font-semibold text-white shadow-xs hover:bg-[#1565EF]/90">
          <Share07 className="size-4" />
          Share
        </button>
      </div>
    </div>
  );
}
