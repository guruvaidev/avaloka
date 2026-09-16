
-- Departments
CREATE TABLE public.departments (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  department text NOT NULL,
  code text NOT NULL,
  head_name text,
  head_email text,
  head_avatar text,
  employees integer NOT NULL DEFAULT 0,
  status text NOT NULL DEFAULT 'Active' CHECK (status IN ('Active','Inactive')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.departments TO authenticated;
GRANT ALL ON public.departments TO service_role;
ALTER TABLE public.departments ENABLE ROW LEVEL SECURITY;
CREATE POLICY "admin all departments" ON public.departments FOR ALL TO authenticated
  USING (public.has_role(auth.uid(), 'admin')) WITH CHECK (public.has_role(auth.uid(), 'admin'));
CREATE TRIGGER set_departments_updated_at BEFORE UPDATE ON public.departments
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- Roles (config)
CREATE TABLE public.config_roles (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  role text NOT NULL,
  department text NOT NULL,
  permissions text[] NOT NULL DEFAULT '{}',
  status text NOT NULL DEFAULT 'Active' CHECK (status IN ('Active','Inactive')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.config_roles TO authenticated;
GRANT ALL ON public.config_roles TO service_role;
ALTER TABLE public.config_roles ENABLE ROW LEVEL SECURITY;
CREATE POLICY "admin all config_roles" ON public.config_roles FOR ALL TO authenticated
  USING (public.has_role(auth.uid(), 'admin')) WITH CHECK (public.has_role(auth.uid(), 'admin'));
CREATE TRIGGER set_config_roles_updated_at BEFORE UPDATE ON public.config_roles
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- Branches
CREATE TABLE public.branches (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  location text NOT NULL,
  head_name text,
  head_email text,
  head_avatar text,
  employees integer NOT NULL DEFAULT 0,
  status text NOT NULL DEFAULT 'Active' CHECK (status IN ('Active','Inactive')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.branches TO authenticated;
GRANT ALL ON public.branches TO service_role;
ALTER TABLE public.branches ENABLE ROW LEVEL SECURITY;
CREATE POLICY "admin all branches" ON public.branches FOR ALL TO authenticated
  USING (public.has_role(auth.uid(), 'admin')) WITH CHECK (public.has_role(auth.uid(), 'admin'));
CREATE TRIGGER set_branches_updated_at BEFORE UPDATE ON public.branches
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- App users (configuration)
CREATE TABLE public.app_users (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  email text NOT NULL,
  avatar text,
  user_id_code text NOT NULL,
  role text NOT NULL,
  department text NOT NULL,
  branch text NOT NULL,
  status text NOT NULL DEFAULT 'Active' CHECK (status IN ('Active','Inactive')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.app_users TO authenticated;
GRANT ALL ON public.app_users TO service_role;
ALTER TABLE public.app_users ENABLE ROW LEVEL SECURITY;
CREATE POLICY "admin all app_users" ON public.app_users FOR ALL TO authenticated
  USING (public.has_role(auth.uid(), 'admin')) WITH CHECK (public.has_role(auth.uid(), 'admin'));
CREATE TRIGGER set_app_users_updated_at BEFORE UPDATE ON public.app_users
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- Teams
CREATE TABLE public.teams (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  lead_name text,
  lead_email text,
  lead_avatar text,
  user_count integer NOT NULL DEFAULT 0,
  user_avatars text[] NOT NULL DEFAULT '{}',
  branch text NOT NULL,
  status text NOT NULL DEFAULT 'Active' CHECK (status IN ('Active','Inactive')),
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.teams TO authenticated;
GRANT ALL ON public.teams TO service_role;
ALTER TABLE public.teams ENABLE ROW LEVEL SECURITY;
CREATE POLICY "admin all teams" ON public.teams FOR ALL TO authenticated
  USING (public.has_role(auth.uid(), 'admin')) WITH CHECK (public.has_role(auth.uid(), 'admin'));
CREATE TRIGGER set_teams_updated_at BEFORE UPDATE ON public.teams
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- Models
CREATE TABLE public.models (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  task_type text NOT NULL,
  tags text[] NOT NULL DEFAULT '{}',
  version text NOT NULL,
  updated_label text NOT NULL DEFAULT '',
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
GRANT SELECT, INSERT, UPDATE, DELETE ON public.models TO authenticated;
GRANT ALL ON public.models TO service_role;
ALTER TABLE public.models ENABLE ROW LEVEL SECURITY;
CREATE POLICY "admin all models" ON public.models FOR ALL TO authenticated
  USING (public.has_role(auth.uid(), 'admin')) WITH CHECK (public.has_role(auth.uid(), 'admin'));
CREATE TRIGGER set_models_updated_at BEFORE UPDATE ON public.models
  FOR EACH ROW EXECUTE FUNCTION public.set_updated_at();

-- Seed data
INSERT INTO public.departments (department, code, head_name, head_email, head_avatar, employees, status) VALUES
('Data Research','DR','Olivia Rhye','olivia@example.com','https://i.pravatar.cc/80?img=47',21,'Active'),
('Data Research','DR','Liam Smith','liam@example.com','https://i.pravatar.cc/80?img=12',35,'Inactive'),
('Data Engineering','DE','Emma Johnson','emma@example.com','https://i.pravatar.cc/80?img=45',29,'Active'),
('Data Researcher','DR','Noah Brown','noah@example.com','https://i.pravatar.cc/80?img=13',43,'Active'),
('Data Engineering','DE','Sophia Davis','sophia@example.com','https://i.pravatar.cc/80?img=32',27,'Inactive'),
('Data Analyst','DA','James Wilson','james@example.com','https://i.pravatar.cc/80?img=14',31,'Active'),
('Data Engineering','DE','Ava Martinez','ava@example.com','https://i.pravatar.cc/80?img=48',24,'Active'),
('Data Scientist','DS','Oliver Garcia','oliver@example.com','https://i.pravatar.cc/80?img=15',38,'Inactive'),
('Data Engineering','DE','Isabella Rodriguez','isabella@example.com','https://i.pravatar.cc/80?img=49',30,'Active'),
('Data Analyst','DA','Elijah Lee','elijah@example.com','https://i.pravatar.cc/80?img=16',26,'Inactive');

INSERT INTO public.config_roles (role, department, permissions, status) VALUES
('Data Researcher','Product Management',ARRAY['Analysis','Projects'],'Active'),
('Data Researcher','Software Engineering',ARRAY['Reports','Cost'],'Inactive'),
('Data Engineering','Human Resources',ARRAY['Revenue','Analysis','Projects','Cost'],'Active'),
('Data Researcher','Marketing',ARRAY['Cost','Revenue','Analysis','Projects','Reports'],'Active'),
('Data Engineering','Sales',ARRAY['Analysis'],'Inactive'),
('Data Analyst','Finance',ARRAY['Revenue'],'Active'),
('Data Engineering','Customer Support',ARRAY['Cost'],'Active'),
('Data Scientist','Legal',ARRAY['Projects'],'Inactive'),
('Data Engineering','Research and Development',ARRAY['Cost'],'Active'),
('Data Analyst','Information Technology',ARRAY['Revenue'],'Inactive');

INSERT INTO public.branches (name, location, head_name, head_email, head_avatar, employees, status) VALUES
('Sector 1','4140 Parker Rd. Allentown, NY','Olivia Rhye','olivia@example.com','https://i.pravatar.cc/80?img=47',21,'Active'),
('Sector 1','4140 Parker Rd. Allentown, NY','Olivia Rhye','olivia@example.com','https://i.pravatar.cc/80?img=47',21,'Active'),
('Sector 1','4140 Parker Rd. Allentown, NY',NULL,NULL,NULL,21,'Inactive'),
('Sector 3','456 Maple Ave. Austin, TX','Emma Johnson','emma@example.com','https://i.pravatar.cc/80?img=45',28,'Active'),
('Sector 1','4140 Parker Rd. Allentown, NY','Olivia Rhye','olivia@example.com','https://i.pravatar.cc/80?img=47',21,'Active'),
('Sector 4','789 Oak Dr. Seattle, WA','Liam Brown','liam@example.com','https://i.pravatar.cc/80?img=12',45,'Inactive'),
('Sector 1','4140 Parker Rd. Allentown, NY','Olivia Rhye','olivia@example.com','https://i.pravatar.cc/80?img=47',21,'Active'),
('Sector 5','101 Pine Ln. Denver, CO','Sophia Garcia','sophia@example.com','https://i.pravatar.cc/80?img=32',38,'Active'),
('Sector 2','123 Elm St. Springfield, IL','Jacob Smith','jacob@example.com','https://i.pravatar.cc/80?img=14',30,'Inactive'),
('Sector 6','202 Birch St. Portland, OR','Noah Wilson','noah@example.com','https://i.pravatar.cc/80?img=13',25,'Active');

INSERT INTO public.app_users (name, email, avatar, user_id_code, role, department, branch, status) VALUES
('Liam Chen','liam@example.com','https://i.pravatar.cc/80?img=12','DA-002','Data Scientist','IT','Bangalore','Active'),
('Liam Chen','liam@example.com','https://i.pravatar.cc/80?img=12','DA-002','Data Scientist','IT','Bangalore','Active'),
('Noah Smith','noah@example.com','https://i.pravatar.cc/80?img=13','DA-004','Data Engineer','IT','Pune','Inactive'),
('Mia Garcia','mia@example.com','https://i.pravatar.cc/80?img=45','DA-010','Systems Analyst','IT','Kolkata','Inactive'),
('Liam Chen','liam@example.com','https://i.pravatar.cc/80?img=12','DA-002','Data Scientist','IT','Bangalore','Active'),
('Lucas Hernandez','lucas@example.com','https://i.pravatar.cc/80?img=14','DA-009','QA Engineer','IT','Jaipur','Active'),
('Liam Chen','liam@example.com','https://i.pravatar.cc/80?img=12','DA-002','Data Scientist','IT','Bangalore','Active'),
('Lucas Hernandez','lucas@example.com','https://i.pravatar.cc/80?img=14','DA-009','QA Engineer','IT','Jaipur','Active'),
('Liam Chen','liam@example.com','https://i.pravatar.cc/80?img=12','DA-002','Data Scientist','IT','Bangalore','Active'),
('Isabella Martinez','isabella@example.com','https://i.pravatar.cc/80?img=47','DA-008','Data Architect','IT','Ahmedabad','Inactive');

INSERT INTO public.teams (name, lead_name, lead_email, lead_avatar, user_count, user_avatars, branch, status) VALUES
('Team 7','Courtney Henry','Courtney@example.com','https://i.pravatar.cc/80?img=32',8,ARRAY['https://i.pravatar.cc/80?img=12','https://i.pravatar.cc/80?img=13','https://i.pravatar.cc/80?img=14'],'Bangalore','Inactive'),
('Team 2','Dianne Russell','Dianne@example.com','https://i.pravatar.cc/80?img=45',10,ARRAY['https://i.pravatar.cc/80?img=12','https://i.pravatar.cc/80?img=13','https://i.pravatar.cc/80?img=14','https://i.pravatar.cc/80?img=15'],'Hyderabad','Active'),
('Team 7','Courtney Henry','Courtney@example.com','https://i.pravatar.cc/80?img=32',8,ARRAY['https://i.pravatar.cc/80?img=12','https://i.pravatar.cc/80?img=13','https://i.pravatar.cc/80?img=14'],'Bangalore','Inactive'),
('Team 4','Jerome Bell','Jerome@example.com','https://i.pravatar.cc/80?img=47',2,ARRAY['https://i.pravatar.cc/80?img=12'],'Mumbai','Inactive'),
('Team 8','Savannah Nguyen','Savannah@example.com','https://i.pravatar.cc/80?img=48',5,ARRAY['https://i.pravatar.cc/80?img=12','https://i.pravatar.cc/80?img=13','https://i.pravatar.cc/80?img=14'],'Pune','Active'),
('Team 3','Marvin McKinney','Marvin@example.com','https://i.pravatar.cc/80?img=13',2,ARRAY['https://i.pravatar.cc/80?img=14'],'Jaipur','Inactive'),
('Team 5','Jacob Jones','Jacob@example.com','https://i.pravatar.cc/80?img=14',2,ARRAY['https://i.pravatar.cc/80?img=12'],'Chennai','Active'),
('Team 9','Eleanor Pena','Eleanor@example.com','https://i.pravatar.cc/80?img=49',10,ARRAY['https://i.pravatar.cc/80?img=12','https://i.pravatar.cc/80?img=13','https://i.pravatar.cc/80?img=14','https://i.pravatar.cc/80?img=15'],'Surat','Active'),
('Team 1','Floyd Miles','Floyd@example.com','https://i.pravatar.cc/80?img=15',3,ARRAY['https://i.pravatar.cc/80?img=12','https://i.pravatar.cc/80?img=13'],'Delhi','Inactive'),
('Team 6','Robert Fox','Robert@example.com','https://i.pravatar.cc/80?img=16',10,ARRAY['https://i.pravatar.cc/80?img=12','https://i.pravatar.cc/80?img=13','https://i.pravatar.cc/80?img=14','https://i.pravatar.cc/80?img=15'],'Kolkata','Active'),
('Team 10','Darlene Robertson','Darlene@example.com','https://i.pravatar.cc/80?img=32',10,ARRAY['https://i.pravatar.cc/80?img=12','https://i.pravatar.cc/80?img=13','https://i.pravatar.cc/80?img=14','https://i.pravatar.cc/80?img=15'],'Ahmedabad','Inactive');

INSERT INTO public.models (name, task_type, tags, version, updated_label) VALUES
('RevenueForecast','Forecasting',ARRAY['Champion'],'V 1.0','2 days ago'),
('Global Sales','Prediction',ARRAY['Forecasting 1','Champion'],'V 2.0','2 days ago'),
('Project Management','Comparison',ARRAY['Comparison','Comparison v2'],'V 3.0','2 days ago'),
('Employee Performance','Forecasting',ARRAY['Label','Label','Label'],'V 5.0','2 days ago'),
('Report 6','Forecasting',ARRAY['Label','Label','Label'],'V 7.0','2 days ago'),
('Quarterly Sales Report','Forecasting',ARRAY['Label','Label','Label'],'V 1.0','2 days ago'),
('Trend Analysis Report','Forecasting',ARRAY['Label','Label','Label'],'V 2.0','2 days ago'),
('Project Management','Forecasting',ARRAY['Label','Label','Label'],'V 4.0','2 days ago'),
('Project Management','Forecasting',ARRAY['Label','Label','Label'],'V 6.0','2 days ago'),
('Project Management','Forecasting',ARRAY['Label','Label','Label'],'V 1.0','2 days ago'),
('Project Management','Forecasting',ARRAY['Label','Label','Label','Label','Label','Label','Label'],'V 2.0','2 days ago');
