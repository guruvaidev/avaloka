import { useEffect, useState } from "react";
import { Calendar, X, HelpCircle, ChevronDown, CheckCircle, User01 } from "@untitledui/icons";
import { Dialog, DialogContent, DialogOverlay, DialogPortal } from "@/components/ui/dialog";
import { Button } from "@/components/base/buttons/button";
import { cx } from "@/lib/utils/cx";

interface ScheduleDeliveryModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  projectName?: string;
  onSchedule?: (data: ScheduleDeliveryData) => void;
}

export interface ScheduleDeliveryData {
  schedule: string;
  from: string;
  to: string;
  time: string;
  user: string;
  selectAll: boolean;
}

const SCHEDULES = ["Daily", "Weekly", "Monthly"];
const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];
const TIMES = [
  "12:00 AM", "1:00 AM", "2:00 AM", "9:00 AM", "10:00 AM",
  "11:00 AM", "12:00 PM", "1:00 PM", "2:00 PM", "5:00 PM",
];
const USERS = ["Olivia Rhye", "Phoenix Baker", "Lana Steiner", "Demi Wilkinson"];

const fieldCx =
  "h-11 w-full rounded-lg border border-secondary bg-primary px-3.5 text-md text-primary placeholder:text-placeholder shadow-xs outline-none focus:border-brand focus:ring-2 focus:ring-brand/20 transition";

function Field({
  label,
  tooltip,
  children,
}: {
  label: string;
  tooltip?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="flex items-center gap-1 text-sm font-semibold text-primary">
        {label}
        {tooltip && <HelpCircle className="size-4 text-fg-quaternary" />}
      </label>
      {children}
    </div>
  );
}

function Select({
  value,
  onChange,
  placeholder,
  options,
  leadingIcon,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  options: string[];
  leadingIcon?: React.ReactNode;
}) {
  return (
    <div className="relative">
      {leadingIcon && (
        <div className="pointer-events-none absolute left-3.5 top-1/2 -translate-y-1/2 text-fg-quaternary">
          {leadingIcon}
        </div>
      )}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={cx(
          fieldCx,
          "appearance-none pr-10",
          leadingIcon && "pl-10",
          !value && "text-placeholder",
        )}
      >
        <option value="" disabled>{placeholder}</option>
        {options.map((o) => (
          <option key={o} value={o} className="text-primary">{o}</option>
        ))}
      </select>
      <ChevronDown className="pointer-events-none absolute right-3.5 top-1/2 size-5 -translate-y-1/2 text-fg-quaternary" />
    </div>
  );
}

export function ScheduleDeliveryModal({
  open,
  onOpenChange,
  projectName = "Project A",
  onSchedule,
}: ScheduleDeliveryModalProps) {
  const [schedule, setSchedule] = useState("Weekly");
  const [from, setFrom] = useState("");
  const [to, setTo] = useState("");
  const [time, setTime] = useState("12:00 AM");
  const [user, setUser] = useState("");
  const [selectAll, setSelectAll] = useState(false);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    if (!open) {
      const t = setTimeout(() => setSuccess(false), 200);
      return () => clearTimeout(t);
    }
  }, [open]);

  useEffect(() => {
    if (!success) return;
    const t = setTimeout(() => onOpenChange(false), 2000);
    return () => clearTimeout(t);
  }, [success, onOpenChange]);

  const handleSubmit = () => {
    onSchedule?.({ schedule, from, to, time, user, selectAll });
    setSuccess(true);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogPortal>
        <DialogOverlay />
        <DialogContent
          className={cx(
            "gap-0 overflow-hidden rounded-2xl border-secondary bg-primary p-0 shadow-xl",
            success ? "max-w-[440px]" : "max-w-[560px]",
          )}
        >
          {success ? (
            <div className="flex flex-col items-center justify-center px-8 py-12 text-center">
              <div className="flex size-14 items-center justify-center rounded-full bg-success-secondary">
                <CheckCircle className="size-7 text-success-primary" />
              </div>
              <h2 className="mt-6 text-lg font-semibold text-primary">
                Successfully Scheduled Report Delivery
              </h2>
              <p className="mt-1 text-md text-tertiary">{projectName}</p>
            </div>
          ) : (
            <>
              {/* Header */}
              <div className="flex items-start justify-between gap-4 border-b border-secondary px-6 py-5">
                <div className="flex items-start gap-4">
                  <div className="flex size-12 shrink-0 items-center justify-center rounded-xl border border-secondary bg-primary shadow-xs">
                    <Calendar className="size-6 text-fg-secondary" />
                  </div>
                  <div>
                    <h2 className="text-lg font-semibold text-primary">Schedule Report Delivery</h2>
                    <p className="mt-0.5 text-sm text-tertiary">
                      You can schedule report delivery to people. Invited colleagues will receive email.
                    </p>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => onOpenChange(false)}
                  className="rounded-md p-1 text-fg-quaternary transition hover:bg-primary_hover hover:text-fg-quaternary_hover"
                  aria-label="Close"
                >
                  <X className="size-5" />
                </button>
              </div>

              {/* Body */}
              <div className="flex flex-col gap-5 px-6 py-5">
                <Field label="Schedule">
                  <Select
                    value={schedule}
                    onChange={setSchedule}
                    placeholder="Select Schedule"
                    options={SCHEDULES}
                  />
                </Field>

                <div className="grid grid-cols-2 gap-4">
                  <Field label="From">
                    <Select
                      value={from}
                      onChange={setFrom}
                      placeholder="Select Start Day"
                      options={DAYS}
                    />
                  </Field>
                  <Field label="To">
                    <Select
                      value={to}
                      onChange={setTo}
                      placeholder="Select End Day"
                      options={DAYS}
                    />
                  </Field>
                </div>

                <Field label="Time">
                  <Select
                    value={time}
                    onChange={setTime}
                    placeholder="12:00 AM"
                    options={TIMES}
                  />
                </Field>

                <Field label="User" tooltip>
                  <Select
                    value={user}
                    onChange={setUser}
                    placeholder="Select User"
                    options={USERS}
                    leadingIcon={<User01 className="size-5" />}
                  />
                </Field>

                <label className="flex cursor-pointer items-center gap-2 text-sm text-secondary">
                  <input
                    type="checkbox"
                    checked={selectAll}
                    onChange={(e) => setSelectAll(e.target.checked)}
                    className="size-4 rounded border-secondary text-brand-solid focus:ring-brand"
                  />
                  Select all members in the project
                </label>
              </div>

              {/* Footer */}
              <div className="flex items-center justify-end gap-3 border-t border-secondary px-6 py-4">
                <Button color="secondary" size="md" onClick={() => onOpenChange(false)}>
                  Cancel
                </Button>
                <Button color="primary" size="md" onClick={handleSubmit}>
                  Schedule
                </Button>
              </div>
            </>
          )}
        </DialogContent>
      </DialogPortal>
    </Dialog>
  );
}
