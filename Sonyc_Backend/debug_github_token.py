import os
import requests
from dotenv import load_dotenv, dotenv_values

def check_github_token():
    print("----- Checking Configuration -----")
    
    # Reload environment
    load_dotenv(override=True)
    
    # Get token
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN") or os.environ.get("GITHUB_ACCESS_TOKEN")
    
    if not token:
        # Try direct .env read
        try:
            env_vals = dotenv_values(".env")
            token = env_vals.get("GITHUB_PERSONAL_ACCESS_TOKEN") or env_vals.get("GITHUB_ACCESS_TOKEN")
        except:
            pass
            
    if not token:
        print("[CRITICAL] No token found!")
        return

    print(f"[OK] Token found (Length: {len(token)})")
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Test 1: User Identity
    print("\n----- Test 1: User Identity -----")
    try:
        resp = requests.get("https://api.github.com/user", headers=headers)
        if resp.status_code == 200:
            user = resp.json()
            print(f"[SUCCESS] Authenticated as: {user.get('login')}")
            print(f"Scopes: {resp.headers.get('X-OAuth-Scopes', 'None')}")
        else:
            print(f"[FAIL] User endpoint returned {resp.status_code}")
            print(resp.text)
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")

    # Test 2: Specific Repository Access
    # Repo: abd027/Sonyc_Frontend
    print("\n----- Test 2: Repository Access (abd027/Sonyc_Frontend) -----")
    target_url = "https://api.github.com/repos/abd027/Sonyc_Frontend/git/trees/main?recursive=1"
    try:
        resp = requests.get(target_url, headers=headers)
        if resp.status_code == 200:
            print("[SUCCESS] Successfully accessed repository tree.")
            data = resp.json()
            print(f"Tree sha: {data.get('sha')}")
            print(f"Truncated: {data.get('truncated')}")
        else:
            print(f"[FAIL] Repo access returned {resp.status_code}")
            print(f"Response: {resp.text}")
            print("Possible reasons: Token missing 'repo' scope, or User does not have access to this repo.")
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")

if __name__ == "__main__":
    check_github_token()
