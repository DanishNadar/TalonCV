create extension if not exists pgcrypto;

create type public.analysis_job_status as enum (
  'queued', 'claimed', 'processing', 'complete', 'failed', 'cancelled'
);

create table public.interview_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  interview_question text not null check (char_length(interview_question) between 1 and 2000),
  target_role text not null default '' check (char_length(target_role) <= 500),
  job_description text not null default '' check (char_length(job_description) <= 10000),
  desired_competencies text not null default '' check (char_length(desired_competencies) <= 2000),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  expires_at timestamptz
);

create table public.recordings (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.interview_sessions(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  storage_path text not null unique check (storage_path !~ '(^/|\\|(^|/)\.\.(/|$))'),
  mime_type text not null,
  file_size_bytes bigint not null check (file_size_bytes > 0 and file_size_bytes <= 262144000),
  duration_seconds numeric check (duration_seconds is null or duration_seconds between 0 and 300.5),
  sha256 text check (sha256 is null or sha256 ~ '^[a-f0-9]{64}$'),
  created_at timestamptz not null default now(),
  unique (id, session_id, user_id)
);

create table public.analysis_jobs (
  id uuid primary key default gen_random_uuid(),
  recording_id uuid not null,
  session_id uuid not null,
  user_id uuid not null,
  status public.analysis_job_status not null default 'queued',
  stage text not null default 'queued',
  progress smallint not null default 0 check (progress between 0 and 100),
  attempt_count smallint not null default 0 check (attempt_count >= 0),
  max_attempts smallint not null default 3 check (max_attempts between 1 and 10),
  claimed_at timestamptz,
  started_at timestamptz,
  completed_at timestamptz,
  cancellation_requested_at timestamptz,
  worker_id text,
  error_code text,
  error_message text,
  analysis_version text not null default 'multimodal-v4',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  foreign key (recording_id, session_id, user_id)
    references public.recordings(id, session_id, user_id) on delete cascade
);

create unique index one_active_job_per_recording
  on public.analysis_jobs(recording_id)
  where status in ('queued', 'claimed', 'processing');
create unique index one_active_job_per_user
  on public.analysis_jobs(user_id)
  where status in ('queued', 'claimed', 'processing');
create index analysis_jobs_claim_queue on public.analysis_jobs(status, created_at);
create index analysis_jobs_user_created on public.analysis_jobs(user_id, created_at desc);
create index analysis_jobs_session_created on public.analysis_jobs(session_id, created_at desc);
create index recordings_user_created on public.recordings(user_id, created_at desc);
create index recordings_session on public.recordings(session_id);
create index interview_sessions_user_created on public.interview_sessions(user_id, created_at desc);
create index interview_sessions_expires on public.interview_sessions(expires_at) where expires_at is not null;

create table public.analysis_results (
  id uuid primary key default gen_random_uuid(),
  job_id uuid not null unique references public.analysis_jobs(id) on delete cascade,
  session_id uuid not null references public.interview_sessions(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  scores jsonb not null default '{}'::jsonb,
  coverage jsonb not null default '{}'::jsonb,
  summary jsonb not null default '{}'::jsonb,
  model_versions jsonb not null default '{}'::jsonb,
  stage_durations jsonb not null default '{}'::jsonb,
  warnings jsonb not null default '[]'::jsonb,
  analysis_version text not null,
  created_at timestamptz not null default now()
);
create index analysis_results_user_created on public.analysis_results(user_id, created_at desc);
create index analysis_results_session on public.analysis_results(session_id);

create table public.artifacts (
  id uuid primary key default gen_random_uuid(),
  job_id uuid not null references public.analysis_jobs(id) on delete cascade,
  session_id uuid not null references public.interview_sessions(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  artifact_type text not null,
  storage_path text not null unique check (storage_path !~ '(^/|\\|(^|/)\.\.(/|$))'),
  content_type text not null,
  size_bytes bigint not null check (size_bytes > 0),
  sha256 text not null check (sha256 ~ '^[a-f0-9]{64}$'),
  created_at timestamptz not null default now(),
  unique (job_id, artifact_type)
);
create index artifacts_user_created on public.artifacts(user_id, created_at desc);
create index artifacts_job on public.artifacts(job_id);
create index artifacts_session on public.artifacts(session_id);

create or replace function public.set_updated_at()
returns trigger language plpgsql set search_path = '' as $$
begin
  new.updated_at = now();
  return new;
end;
$$;
create trigger interview_sessions_updated before update on public.interview_sessions
  for each row execute function public.set_updated_at();
create trigger analysis_jobs_updated before update on public.analysis_jobs
  for each row execute function public.set_updated_at();

alter table public.interview_sessions enable row level security;
alter table public.recordings enable row level security;
alter table public.analysis_jobs enable row level security;
alter table public.analysis_results enable row level security;
alter table public.artifacts enable row level security;

create policy "sessions_select_own" on public.interview_sessions for select to authenticated
  using ((select auth.uid()) = user_id);
create policy "sessions_insert_own" on public.interview_sessions for insert to authenticated
  with check ((select auth.uid()) = user_id);
create policy "sessions_update_own" on public.interview_sessions for update to authenticated
  using ((select auth.uid()) = user_id) with check ((select auth.uid()) = user_id);
create policy "sessions_delete_own" on public.interview_sessions for delete to authenticated
  using ((select auth.uid()) = user_id);

create policy "recordings_select_own" on public.recordings for select to authenticated
  using ((select auth.uid()) = user_id);
create policy "recordings_insert_own" on public.recordings for insert to authenticated
  with check (
    (select auth.uid()) = user_id and exists (
      select 1 from public.interview_sessions s where s.id = session_id and s.user_id = (select auth.uid())
    )
  );

create policy "jobs_select_own" on public.analysis_jobs for select to authenticated
  using ((select auth.uid()) = user_id);
create policy "results_select_own" on public.analysis_results for select to authenticated
  using ((select auth.uid()) = user_id);
create policy "artifacts_select_own" on public.artifacts for select to authenticated
  using ((select auth.uid()) = user_id);

revoke all on table public.interview_sessions, public.recordings, public.analysis_jobs,
  public.analysis_results, public.artifacts from anon, authenticated;
grant select, insert, update, delete on table public.interview_sessions to authenticated;
grant select, insert on table public.recordings to authenticated;
grant select on table public.analysis_jobs, public.analysis_results, public.artifacts to authenticated;

create or replace function public.enqueue_analysis_job(p_recording_id uuid)
returns public.analysis_jobs
language plpgsql
security definer
set search_path = public
as $$
declare
  v_user uuid := auth.uid();
  v_recording public.recordings;
  v_existing public.analysis_jobs;
  v_job public.analysis_jobs;
  v_is_anonymous boolean := coalesce((auth.jwt()->>'is_anonymous')::boolean, false);
begin
  if v_user is null then raise exception 'authentication_required'; end if;
  select * into v_recording from public.recordings where id = p_recording_id and user_id = v_user;
  if not found then raise exception 'recording_not_found'; end if;
  select * into v_existing from public.analysis_jobs
    where recording_id = p_recording_id and status in ('queued', 'claimed', 'processing')
    order by created_at desc limit 1;
  if found then return v_existing; end if;
  if exists(select 1 from public.analysis_jobs where user_id = v_user and status in ('queued', 'claimed', 'processing')) then
    raise exception 'active_job_limit';
  end if;
  if v_is_anonymous and (
    select count(*) from public.analysis_jobs where user_id = v_user and created_at > now() - interval '24 hours'
  ) >= 3 then
    raise exception 'guest_daily_limit';
  end if;
  begin
    insert into public.analysis_jobs(recording_id, session_id, user_id)
      values (v_recording.id, v_recording.session_id, v_user) returning * into v_job;
  exception when unique_violation then
    select * into v_existing from public.analysis_jobs
      where recording_id = p_recording_id and status in ('queued', 'claimed', 'processing')
      order by created_at desc limit 1;
    if found then return v_existing; end if;
    if exists(select 1 from public.analysis_jobs where user_id = v_user and status in ('queued', 'claimed', 'processing')) then
      raise exception 'active_job_limit';
    end if;
    raise;
  end;
  return v_job;
end;
$$;
revoke all on function public.enqueue_analysis_job(uuid) from public;
grant execute on function public.enqueue_analysis_job(uuid) to authenticated;

create or replace function public.cancel_analysis_job(p_job_id uuid)
returns public.analysis_jobs
language plpgsql
security definer
set search_path = public
as $$
declare v_job public.analysis_jobs;
begin
  update public.analysis_jobs set
      status=case when status in ('queued','claimed') then 'cancelled'::public.analysis_job_status else status end,
      stage=case when status in ('queued','claimed') then 'cancelled' else 'cancellation_requested' end,
      cancellation_requested_at=now(),
      completed_at=case when status in ('queued','claimed') then now() else completed_at end,
      worker_id=case when status in ('queued','claimed') then null else worker_id end,
      claimed_at=case when status in ('queued','claimed') then null else claimed_at end,
      updated_at=now()
    where id=p_job_id and user_id=auth.uid() and status in ('queued','claimed','processing')
    returning * into v_job;
  if not found then raise exception 'job_not_cancellable'; end if;
  return v_job;
end;
$$;
revoke all on function public.cancel_analysis_job(uuid) from public;
grant execute on function public.cancel_analysis_job(uuid) to authenticated;

create or replace function public.claim_analysis_job(p_worker_id text)
returns setof public.analysis_jobs
language plpgsql
security definer
set search_path = public
as $$
begin
  return query
  with candidate as (
    select id from public.analysis_jobs where status='queued' and cancellation_requested_at is null
    order by created_at for update skip locked limit 1
  )
  update public.analysis_jobs j set
    status='claimed', stage='preparing_media', progress=1, worker_id=p_worker_id,
    claimed_at=now(), started_at=coalesce(j.started_at, now()), attempt_count=j.attempt_count+1,
    error_code=null, error_message=null, updated_at=now()
  from candidate where j.id=candidate.id returning j.*;
end;
$$;
revoke all on function public.claim_analysis_job(text) from public, anon, authenticated;
grant execute on function public.claim_analysis_job(text) to service_role;

create or replace function public.recover_stale_analysis_jobs(p_stale_minutes integer default 20)
returns integer language plpgsql security definer set search_path = public as $$
declare v_count integer;
begin
  with changed as (
    update public.analysis_jobs set
      status=case
        when cancellation_requested_at is not null then 'cancelled'::public.analysis_job_status
        when attempt_count < max_attempts then 'queued'::public.analysis_job_status
        else 'failed'::public.analysis_job_status end,
      stage=case
        when cancellation_requested_at is not null then 'cancelled'
        when attempt_count < max_attempts then 'queued'
        else 'failed' end,
      worker_id=null, claimed_at=null, error_code='worker_stale',
      error_message='The analysis worker stopped responding and the job was recovered.',
      completed_at=case when cancellation_requested_at is not null or attempt_count >= max_attempts then now() else completed_at end,
      updated_at=now()
    where status in ('claimed','processing') and updated_at < now() - make_interval(mins => greatest(p_stale_minutes, 5))
    returning 1
  ) select count(*) into v_count from changed;
  return v_count;
end;
$$;
revoke all on function public.recover_stale_analysis_jobs(integer) from public, anon, authenticated;
grant execute on function public.recover_stale_analysis_jobs(integer) to service_role;

create or replace function public.complete_analysis_job(p_job_id uuid, p_result jsonb, p_artifacts jsonb)
returns void language plpgsql security definer set search_path = public as $$
declare v_job public.analysis_jobs;
begin
  select * into v_job from public.analysis_jobs where id=p_job_id for update;
  if not found or v_job.status not in ('claimed','processing') then raise exception 'invalid_job_state'; end if;
  insert into public.analysis_results(
    job_id,session_id,user_id,scores,coverage,summary,model_versions,stage_durations,warnings,analysis_version
  ) values (
    v_job.id,v_job.session_id,v_job.user_id,
    coalesce(p_result->'scores','{}'),coalesce(p_result->'coverage','{}'),coalesce(p_result->'summary','{}'),
    coalesce(p_result->'model_versions','{}'),coalesce(p_result->'stage_durations','{}'),coalesce(p_result->'warnings','[]'),
    coalesce(p_result->>'analysis_version',v_job.analysis_version)
  ) on conflict(job_id) do update set
    scores=excluded.scores,coverage=excluded.coverage,summary=excluded.summary,
    model_versions=excluded.model_versions,stage_durations=excluded.stage_durations,warnings=excluded.warnings;
  insert into public.artifacts(job_id,session_id,user_id,artifact_type,storage_path,content_type,size_bytes,sha256)
    select v_job.id,v_job.session_id,v_job.user_id,x.artifact_type,x.storage_path,x.content_type,x.size_bytes,x.sha256
    from jsonb_to_recordset(coalesce(p_artifacts,'[]')) as x(
      artifact_type text,storage_path text,content_type text,size_bytes bigint,sha256 text
    ) on conflict(job_id,artifact_type) do update set
      storage_path=excluded.storage_path,content_type=excluded.content_type,size_bytes=excluded.size_bytes,sha256=excluded.sha256;
  update public.analysis_jobs set
    status='complete',stage='complete',progress=100,completed_at=now(),cancellation_requested_at=null,updated_at=now()
    where id=p_job_id and status in ('claimed','processing');
end;
$$;
revoke all on function public.complete_analysis_job(uuid,jsonb,jsonb) from public, anon, authenticated;
grant execute on function public.complete_analysis_job(uuid,jsonb,jsonb) to service_role;

create or replace function public.fail_analysis_job(
  p_job_id uuid,p_error_code text,p_error_message text,p_recoverable boolean default true
) returns void language plpgsql security definer set search_path = public as $$
declare v_job public.analysis_jobs;
begin
  select * into v_job from public.analysis_jobs where id=p_job_id for update;
  if not found then return; end if;
  update public.analysis_jobs set
    status=case
      when v_job.cancellation_requested_at is not null then 'cancelled'::public.analysis_job_status
      when p_recoverable and v_job.attempt_count < v_job.max_attempts then 'queued'::public.analysis_job_status
      else 'failed'::public.analysis_job_status end,
    stage=case
      when v_job.cancellation_requested_at is not null then 'cancelled'
      when p_recoverable and v_job.attempt_count < v_job.max_attempts then 'queued'
      else 'failed' end,
    progress=case when v_job.cancellation_requested_at is null and p_recoverable and v_job.attempt_count < v_job.max_attempts then 0 else progress end,
    completed_at=case when v_job.cancellation_requested_at is not null or not p_recoverable or v_job.attempt_count >= v_job.max_attempts then now() else completed_at end,
    worker_id=null,claimed_at=null,error_code=left(p_error_code,100),error_message=left(p_error_message,500),updated_at=now()
    where id=p_job_id and status in ('claimed','processing');
end;
$$;
revoke all on function public.fail_analysis_job(uuid,text,text,boolean) from public, anon, authenticated;
grant execute on function public.fail_analysis_job(uuid,text,text,boolean) to service_role;

create or replace function public.release_analysis_job(p_job_id uuid,p_worker_id text)
returns void language sql security definer set search_path = public as $$
  update public.analysis_jobs set
    status=case when cancellation_requested_at is not null then 'cancelled'::public.analysis_job_status else 'queued'::public.analysis_job_status end,
    stage=case when cancellation_requested_at is not null then 'cancelled' else 'queued' end,
    progress=case when cancellation_requested_at is not null then progress else 0 end,
    completed_at=case when cancellation_requested_at is not null then now() else completed_at end,
    worker_id=null,claimed_at=null,updated_at=now()
  where id=p_job_id and worker_id=p_worker_id and status in ('claimed','processing');
$$;
revoke all on function public.release_analysis_job(uuid,text) from public, anon, authenticated;
grant execute on function public.release_analysis_job(uuid,text) to service_role;

create or replace function public.acknowledge_analysis_job_cancellation(p_job_id uuid,p_worker_id text)
returns void language sql security definer set search_path = public as $$
  update public.analysis_jobs set
    status='cancelled',stage='cancelled',completed_at=now(),worker_id=null,claimed_at=null,updated_at=now()
  where id=p_job_id and worker_id=p_worker_id and status in ('claimed','processing') and cancellation_requested_at is not null;
$$;
revoke all on function public.acknowledge_analysis_job_cancellation(uuid,text) from public, anon, authenticated;
grant execute on function public.acknowledge_analysis_job_cancellation(uuid,text) to service_role;

insert into storage.buckets(id,name,public,file_size_limit,allowed_mime_types)
values
  ('taloncv-recordings','taloncv-recordings',false,262144000,array['video/webm','video/mp4','video/quicktime','video/x-matroska','video/x-msvideo','audio/webm','audio/wav','audio/mpeg','audio/mp4','audio/x-m4a','audio/aac','audio/ogg','audio/flac','audio/x-flac']),
  ('taloncv-artifacts','taloncv-artifacts',false,262144000,null)
on conflict(id) do update set public=false,file_size_limit=excluded.file_size_limit,allowed_mime_types=excluded.allowed_mime_types;

create policy "recording_objects_insert_own" on storage.objects for insert to authenticated
  with check (bucket_id='taloncv-recordings' and (storage.foldername(name))[1]='users' and (storage.foldername(name))[2]=(select auth.uid())::text);
create policy "recording_objects_select_own" on storage.objects for select to authenticated
  using (bucket_id='taloncv-recordings' and (storage.foldername(name))[1]='users' and (storage.foldername(name))[2]=(select auth.uid())::text);
create policy "recording_objects_delete_own" on storage.objects for delete to authenticated
  using (bucket_id='taloncv-recordings' and (storage.foldername(name))[1]='users' and (storage.foldername(name))[2]=(select auth.uid())::text);
create policy "artifact_objects_select_own" on storage.objects for select to authenticated
  using (bucket_id='taloncv-artifacts' and (storage.foldername(name))[1]='users' and (storage.foldername(name))[2]=(select auth.uid())::text);

do $$ begin
  alter publication supabase_realtime add table public.analysis_jobs;
exception when duplicate_object then null;
end $$;
