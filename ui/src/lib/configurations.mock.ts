// Mock data for configurations page (no backend tables yet)

export const MOCK_DEPARTMENTS = [
  { id: "d1", department: "Finance", code: "FIN-01", head_name: "Olivia Rhye", head_email: "olivia@avaloka.ai", head_avatar: null, employees: 24, status: "Active" as const },
  { id: "d2", department: "Engineering", code: "ENG-02", head_name: "Phoenix Baker", head_email: "phoenix@avaloka.ai", head_avatar: null, employees: 58, status: "Active" as const },
  { id: "d3", department: "Marketing", code: "MKT-03", head_name: "Lana Steiner", head_email: "lana@avaloka.ai", head_avatar: null, employees: 16, status: "Active" as const },
  { id: "d4", department: "Operations", code: "OPS-04", head_name: "Demi Wilkinson", head_email: "demi@avaloka.ai", head_avatar: null, employees: 32, status: "Inactive" as const },
  { id: "d5", department: "Sales", code: "SAL-05", head_name: "Candice Wu", head_email: "candice@avaloka.ai", head_avatar: null, employees: 41, status: "Active" as const },
];

export const MOCK_ROLES = [
  { id: "r1", role: "Administrator", department: "Engineering", permissions: ["Analysis", "Projects", "Reports", "Cost", "Revenue"], status: "Active" as const },
  { id: "r2", role: "Analyst", department: "Finance", permissions: ["Analysis", "Reports"], status: "Active" as const },
  { id: "r3", role: "Manager", department: "Sales", permissions: ["Projects", "Revenue"], status: "Active" as const },
  { id: "r4", role: "Viewer", department: "Marketing", permissions: ["Reports"], status: "Inactive" as const },
  { id: "r5", role: "Auditor", department: "Operations", permissions: ["Cost", "Reports"], status: "Active" as const },
];

export const MOCK_BRANCHES = [
  { id: "b1", name: "Headquarters", location: "New York, USA", head_name: "Olivia Rhye", head_email: "olivia@avaloka.ai", head_avatar: null, employees: 120, status: "Active" as const },
  { id: "b2", name: "EU Office", location: "Berlin, Germany", head_name: "Phoenix Baker", head_email: "phoenix@avaloka.ai", head_avatar: null, employees: 48, status: "Active" as const },
  { id: "b3", name: "APAC Office", location: "Singapore", head_name: "Lana Steiner", head_email: "lana@avaloka.ai", head_avatar: null, employees: 36, status: "Active" as const },
  { id: "b4", name: "India Dev Center", location: "Bangalore, India", head_name: "Candice Wu", head_email: "candice@avaloka.ai", head_avatar: null, employees: 92, status: "Active" as const },
  { id: "b5", name: "Remote Hub", location: "Distributed", head_name: null, head_email: null, head_avatar: null, employees: 14, status: "Inactive" as const },
];

export const MOCK_APP_USERS = [
  { id: "u1", name: "Olivia Rhye", email: "olivia@avaloka.ai", avatar: null, user_id_code: "USR-001", role: "Administrator", department: "Engineering", branch: "Headquarters", status: "Active" as const },
  { id: "u2", name: "Phoenix Baker", email: "phoenix@avaloka.ai", avatar: null, user_id_code: "USR-002", role: "Manager", department: "Sales", branch: "EU Office", status: "Active" as const },
  { id: "u3", name: "Lana Steiner", email: "lana@avaloka.ai", avatar: null, user_id_code: "USR-003", role: "Analyst", department: "Finance", branch: "APAC Office", status: "Active" as const },
  { id: "u4", name: "Demi Wilkinson", email: "demi@avaloka.ai", avatar: null, user_id_code: "USR-004", role: "Viewer", department: "Marketing", branch: "Headquarters", status: "Inactive" as const },
  { id: "u5", name: "Candice Wu", email: "candice@avaloka.ai", avatar: null, user_id_code: "USR-005", role: "Auditor", department: "Operations", branch: "India Dev Center", status: "Active" as const },
];

export const MOCK_TEAMS = [
  { id: "t1", name: "Data Platform", description: "Owns warehouse & pipelines", lead_user_id: null, lead_name: "Phoenix Baker", lead_email: "phoenix@avaloka.ai", lead_avatar: null, user_count: 8, user_avatars: [], member_ids: [], branch: "Headquarters", status: "Active" as const },
  { id: "t2", name: "Growth", description: "Marketing & demand gen", lead_user_id: null, lead_name: "Lana Steiner", lead_email: "lana@avaloka.ai", lead_avatar: null, user_count: 5, user_avatars: [], member_ids: [], branch: "EU Office", status: "Active" as const },
  { id: "t3", name: "Finance Ops", description: "Close, reporting, audit", lead_user_id: null, lead_name: "Candice Wu", lead_email: "candice@avaloka.ai", lead_avatar: null, user_count: 6, user_avatars: [], member_ids: [], branch: "APAC Office", status: "Active" as const },
  { id: "t4", name: "Customer Success", description: "Onboarding & support", lead_user_id: null, lead_name: "Olivia Rhye", lead_email: "olivia@avaloka.ai", lead_avatar: null, user_count: 12, user_avatars: [], member_ids: [], branch: "India Dev Center", status: "Inactive" as const },
];

