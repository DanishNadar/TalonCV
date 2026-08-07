# Supabase setup

The migration creates user-owned interview tables, atomic job functions, RLS policies, private storage buckets, size/MIME restrictions, and Realtime publication for `analysis_jobs`.

From the repository root:

```bash
npx supabase login
npx supabase link --project-ref YOUR_PROJECT_REF
npx supabase db push
```

For local development:

```bash
npx supabase start
npx supabase db reset
```

Enable anonymous sign-ins under **Authentication → Providers → Anonymous Sign-Ins** for guest mode. Configure CAPTCHA before a wider public beta. If anonymous auth remains disabled, the UI preserves email magic-link sign-in and reports that guest access is unavailable.

Verify after migration:

1. `taloncv-recordings` and `taloncv-artifacts` are private.
2. An authenticated user can access only `users/{their_user_id}/...` objects.
3. A user can select only their own rows.
4. Direct inserts/updates to `analysis_jobs` are denied; `enqueue_analysis_job` and `cancel_analysis_job` are allowed.
5. The `service_role` can call worker RPCs.

Never delete rows directly from `storage.objects`; use the Storage API so the underlying object is removed. Run retention cleanup with:

```bash
python scripts/cleanupExpiredInterviews.py --dry-run
python scripts/cleanupExpiredInterviews.py
```

Schedule the non-dry command from a trusted cron host with server-only Supabase variables. Guest sessions created by the web app default to a seven-day expiry.
