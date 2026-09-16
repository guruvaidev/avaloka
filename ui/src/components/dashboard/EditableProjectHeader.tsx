import { useState } from "react";
import { Check, X } from "@untitledui/icons";
import { ButtonUtility } from "@/components/base/buttons/button-utility";
import { InviteCollaboratorsTrigger } from "./InviteCollaboratorsTrigger";

interface EditableProjectHeaderProps {
  defaultName?: string;
  defaultDescription?: string;
  projectId?: string | null;
  onNameCommit?: (name: string) => void;
  onDescriptionCommit?: (description: string) => void;
  readOnly?: boolean;
}

type FieldState = {
  value: string;
  editing: boolean;
  draft: string;
};

export function EditableProjectHeader({
  defaultName = "",
  defaultDescription = "",
  projectId = null,
  onNameCommit,
  onDescriptionCommit,
  readOnly = false,
}: EditableProjectHeaderProps) {
  const [name, setName] = useState<FieldState>({
    value: defaultName,
    editing: !readOnly,
    draft: defaultName,
  });
  const [desc, setDesc] = useState<FieldState>({
    value: defaultDescription,
    editing: !readOnly,
    draft: defaultDescription,
  });

  if (readOnly) {
    return (
      <div className="relative flex w-full items-start justify-between gap-4 border-b border-secondary px-2 pb-5">
        <div className="flex-1 space-y-1">
          <div className="text-md font-semibold text-primary">
            {defaultName || "Untitled project"}
          </div>
          {defaultDescription ? (
            <div className="text-sm text-tertiary">{defaultDescription}</div>
          ) : null}
        </div>
      </div>
    );
  }

  const commitName = (state: FieldState) => {
    const v = state.draft.trim();
    setName({ value: v, draft: v, editing: false });
    if (v && v !== state.value) onNameCommit?.(v);
  };
  const commitDesc = (state: FieldState) => {
    const v = state.draft.trim();
    setDesc({ value: v, draft: v, editing: false });
    if (v !== state.value) onDescriptionCommit?.(v);
  };

  const cancel = (state: FieldState, setter: (s: FieldState) => void) =>
    setter({ value: state.value, draft: state.value, editing: false });

  const startEdit = (state: FieldState, setter: (s: FieldState) => void) =>
    setter({ ...state, draft: state.value, editing: true });

  return (
    <div className="relative flex w-full items-start justify-between gap-4 border-b border-secondary px-2 pb-5">
      <div className="flex-1 space-y-1">
        {name.editing ? (
          <div className="flex items-center gap-2">
            <input
              autoFocus
              value={name.draft}
              onChange={(e) => setName({ ...name, draft: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitName(name);
                if (e.key === "Escape") cancel(name, setName);
              }}
              onBlur={() => commitName(name)}
              placeholder="Add Project Name"
              className="flex-1 bg-transparent text-md font-semibold text-primary placeholder:text-primary outline-none"
            />
            <ButtonUtility size="xs" color="tertiary" icon={Check} tooltip="Save" onClick={() => commitName(name)} />
            <ButtonUtility size="xs" color="tertiary" icon={X} tooltip="Cancel" onClick={() => cancel(name, setName)} />
          </div>
        ) : (
          <button
            type="button"
            onClick={() => startEdit(name, setName)}
            className="block w-full cursor-text text-left text-md font-semibold text-primary outline-none"
          >
            {name.value || "Add Project Name"}
          </button>
        )}

        {desc.editing ? (
          <div className="flex items-center gap-2">
            <input
              value={desc.draft}
              onChange={(e) => setDesc({ ...desc, draft: e.target.value })}
              onKeyDown={(e) => {
                if (e.key === "Enter") commitDesc(desc);
                if (e.key === "Escape") cancel(desc, setDesc);
              }}
              onBlur={() => commitDesc(desc)}
              placeholder="Add Description"
              className="flex-1 bg-transparent text-sm text-tertiary placeholder:text-tertiary outline-none"
            />
            <ButtonUtility size="xs" color="tertiary" icon={Check} tooltip="Save" onClick={() => commitDesc(desc)} />
            <ButtonUtility size="xs" color="tertiary" icon={X} tooltip="Cancel" onClick={() => cancel(desc, setDesc)} />
          </div>
        ) : (
          <button
            type="button"
            onClick={() => startEdit(desc, setDesc)}
            className="block w-full cursor-text text-left text-sm text-tertiary outline-none"
          >
            {desc.value || "Add Description"}
          </button>
        )}
      </div>

      {!readOnly && <InviteCollaboratorsTrigger projectId={projectId} />}
    </div>
  );
}
