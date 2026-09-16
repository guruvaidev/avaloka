/**
 * Shared (client + server safe) validation for the billing details the user
 * types before checkout, plus the recurring-payment (auto-pay) mandate text.
 *
 * Both the UI forms and the server functions run the SAME rules, so a user can
 * never reach Stripe/PayPal with junk company / address / VAT values, nor
 * without having explicitly authorized automatic recurring charges.
 */

export type BillingDetailsInput = {
  company_name?: string | null;
  billing_address?: string | null;
  vat_number?: string | null;
  auto_pay_authorized?: boolean | null;
};

export type BillingDetailsValues = {
  company_name: string;
  billing_address: string;
  vat_number: string;
  auto_pay_authorized: boolean;
};

export type BillingFieldErrors = Partial<
  Record<keyof BillingDetailsValues, string>
>;

/** Shown next to the mandate checkbox in every checkout / change-method form. */
export const AUTO_PAY_CONSENT_TEXT =
  "I authorize Avaloka to automatically charge this payment method on a recurring basis for the selected plan (after the 15-day free trial) until I cancel.";

const LETTER = /\p{L}/u;
const DIGIT = /[0-9]/;

export function normalizeVat(raw: string): string {
  return raw.replace(/[\s.\-_/]/g, "").toUpperCase();
}

/**
 * Validate + normalize. Returns cleaned values and a per-field error map.
 * `requireAuthorization` defaults to true — the auto-pay mandate is mandatory.
 */
export function validateBillingDetails(
  input: BillingDetailsInput | null | undefined,
  opts: { requireAuthorization?: boolean } = {},
): { values: BillingDetailsValues; errors: BillingFieldErrors } {
  const requireAuthorization = opts.requireAuthorization !== false;
  const errors: BillingFieldErrors = {};

  const company_name = String(input?.company_name ?? "").trim().replace(/\s+/g, " ");
  const billing_address = String(input?.billing_address ?? "").trim().replace(/\s+/g, " ");
  const vatRaw = String(input?.vat_number ?? "").trim();
  const vat_number = normalizeVat(vatRaw);
  const auto_pay_authorized = input?.auto_pay_authorized === true;

  // Company name
  if (!company_name) {
    errors.company_name = "Company name is required.";
  } else if (company_name.length < 2 || company_name.length > 100) {
    errors.company_name = "Company name must be between 2 and 100 characters.";
  } else if (!LETTER.test(company_name)) {
    errors.company_name = "Company name must contain letters.";
  }

  // Billing address — needs a street/number and a place, so require both a
  // digit and letters and a reasonable length.
  if (!billing_address) {
    errors.billing_address = "Billing address is required.";
  } else if (billing_address.length < 10) {
    errors.billing_address = "Enter the full address (street, city, state, ZIP).";
  } else if (billing_address.length > 200) {
    errors.billing_address = "Billing address is too long (max 200 characters).";
  } else if (!LETTER.test(billing_address) || !DIGIT.test(billing_address)) {
    errors.billing_address =
      "Address must include a street number / ZIP as well as the city.";
  }

  // VAT / tax id — alphanumeric, 6–20 chars once separators are stripped.
  if (!vat_number) {
    errors.vat_number = "VAT number is required.";
  } else if (!/^[A-Z0-9]+$/.test(vat_number)) {
    errors.vat_number = "VAT number may only contain letters and digits.";
  } else if (vat_number.length < 6 || vat_number.length > 20) {
    errors.vat_number = "VAT number must be 6 to 20 characters.";
  } else if (!DIGIT.test(vat_number)) {
    errors.vat_number = "VAT number must contain at least one digit.";
  }

  if (requireAuthorization && !auto_pay_authorized) {
    errors.auto_pay_authorized =
      "You must authorize automatic recurring payments to continue.";
  }

  return {
    values: { company_name, billing_address, vat_number, auto_pay_authorized },
    errors,
  };
}

/** Server-side helper: validate or throw the first readable error. */
export function assertBillingDetails(
  input: BillingDetailsInput | null | undefined,
  opts: { requireAuthorization?: boolean } = {},
): BillingDetailsValues {
  const { values, errors } = validateBillingDetails(input, opts);
  const first = Object.values(errors)[0];
  if (first) throw new Error(first);
  return values;
}

/** Server-side helper for flows that only need the mandate (no billing form). */
export function assertAutoPayAuthorized(authorized: unknown): true {
  if (authorized !== true) {
    throw new Error("You must authorize automatic recurring payments to continue.");
  }
  return true;
}
