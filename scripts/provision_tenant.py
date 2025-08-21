#!/usr/bin/env python3
"""
Provision a new Supabase project for a tenant using the Supabase Management API,
then store the credentials in the control plane via the audio-analysis API.

Environment:
  SUPABASE_PAT            - Supabase personal access token with org admin access
  SUPABASE_ORG_ID         - Target organization ID
  CONTROL_PLANE_API_URL   - Base URL of audio-analysis API (e.g., https://api.speechanalyser.com)

Usage:
  python scripts/provision_tenant.py \
    --tenant-id <uuid> \
    --subdomain lakeview \
    --display-name "Lakeview High School" \
    --region us-east-1

Notes:
  - This script is intended to run in CI (GitHub Actions) with secrets configured.
  - Keys never touch the browser/UI; they are stored server-side via the control plane.
"""
import argparse
import os
import sys
import time
import json
from typing import Tuple
import requests


SUPABASE_API = os.getenv("SUPABASE_API_BASE", "https://api.supabase.com")


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        print(f"Missing env: {name}", file=sys.stderr)
        sys.exit(2)
    return val


def create_project(pat: str, org_id: str, name: str, db_password: str, region: str) -> Tuple[str, str]:
    """Create Supabase project; return (project_id, project_ref)."""
    url = f"{SUPABASE_API}/v1/projects"
    headers = {"Authorization": f"Bearer {pat}", "Content-Type": "application/json"}
    payload = {
        "organization_id": org_id,
        "name": name,
        "db_pass": db_password,  # field name expected by Supabase Management API
        "region": region,
        # Optional: "plan": "free"|"pro"
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code not in (200, 201, 202):
        raise RuntimeError(f"Create project failed: {resp.status_code} {resp.text}")
    data = resp.json()
    # Supabase returns an "id" which is also the project ref used in hostnames
    project_id = data.get("id") or data.get("project_id")
    project_ref = data.get("ref") or data.get("reference_id") or data.get("project_ref") or project_id
    if not project_id or not project_ref:
        raise RuntimeError(f"Unexpected create response: {data}")
    return project_id, project_ref


def get_api_keys(pat: str, project_ref: str) -> Tuple[str, str, str]:
    """Return (supabase_url, anon_key, service_role_key)."""
    headers = {"Authorization": f"Bearer {pat}"}
    # Project details
    proj = requests.get(f"{SUPABASE_API}/v1/projects/{project_ref}", headers=headers, timeout=60)
    if proj.status_code != 200:
        raise RuntimeError(f"Get project failed: {proj.status_code} {proj.text}")
    p = proj.json()
    supabase_url = p.get("api_url") or p.get("api") or p.get("endpoint")
    # Keys
    keys = requests.get(f"{SUPABASE_API}/v1/projects/{project_ref}/api-keys", headers=headers, timeout=60)
    if keys.status_code != 200:
        raise RuntimeError(f"Get api-keys failed: {keys.status_code} {keys.text}")
    kd = keys.json()
    anon_key = None
    service_role_key = None
    for item in kd:
        if item.get("name") == "anon" or item.get("role") == "anon":
            anon_key = item.get("api_key") or item.get("key")
        if item.get("name") == "service_role" or item.get("role") == "service_role":
            service_role_key = item.get("api_key") or item.get("key")
    if not (supabase_url and anon_key and service_role_key):
        raise RuntimeError(f"Could not resolve keys: {json.dumps(kd)[:200]}")
    return supabase_url, anon_key, service_role_key


def store_creds(api_base: str, tenant_id: str, supabase_url: str, anon_key: str, service_role_key: str, region: str) -> None:
    url = f"{api_base.rstrip('/')}/api/admin/tenants/{tenant_id}/db"
    resp = requests.post(url, json={
        "supabase_url": supabase_url,
        "anon_key": anon_key,
        "service_role_key": service_role_key,
        "region": region,
    }, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Store creds failed: {resp.status_code} {resp.text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", required=True)
    parser.add_argument("--subdomain", required=True)
    parser.add_argument("--display-name", required=True)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--db-password", default=os.getenv("DEFAULT_TENANT_DB_PASSWORD", "ChangeMe_12345"))
    args = parser.parse_args()

    pat = _require_env("SUPABASE_PAT")
    org = _require_env("SUPABASE_ORG_ID")
    api_base = _require_env("CONTROL_PLANE_API_URL")

    print(f"Creating Supabase project for tenant {args.subdomain} in org {org}...")
    project_id, project_ref = create_project(pat, org, args.display_name, args.db_password, args.region)
    print(f"Project created: {project_id} ({project_ref}) - waiting for API keys to be ready...")
    # Expose project_ref to subsequent workflow steps
    gh_out = os.getenv("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a") as f:
            f.write(f"project_ref={project_ref}\n")
    time.sleep(5)
    supabase_url, anon_key, service_role_key = get_api_keys(pat, project_ref)
    print("Storing credentials in control plane...")
    store_creds(api_base, args.tenant_id, supabase_url, anon_key, service_role_key, args.region)
    print("✅ Tenant provisioned and credentials stored.")


if __name__ == "__main__":
    main()


