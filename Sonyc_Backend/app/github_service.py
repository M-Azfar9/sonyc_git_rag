"""
GitHub Service Module
Handles all GitHub API interactions: fetching repo context (code, issues, PRs, commits, branches),
webhook registration, and webhook signature verification.
"""

import hashlib
import hmac
import logging
import os
import secrets
from typing import Optional

from github import Github, GithubException
from langchain_community.document_loaders import GithubFileLoader

logger = logging.getLogger(__name__)


def _get_github_client(token: str) -> Github:
    """Create a PyGithub client from a token."""
    return Github(token)


def _parse_repo_id(repo_url: str) -> str:
    """Convert a GitHub URL to owner/repo format."""
    cleaned = repo_url.replace("https://", "").replace("http://", "").rstrip("/")
    parts = cleaned.split("/")
    if len(parts) < 3:
        raise ValueError(f"Invalid GitHub URL format: {repo_url}")
    owner = parts[1]
    repo = parts[2].replace(".git", "")
    return f"{owner}/{repo}"


def _parse_owner_and_name(repo_url: str) -> tuple[str, str]:
    """Extract owner and repo name from URL."""
    repo_id = _parse_repo_id(repo_url)
    owner, name = repo_id.split("/")
    return owner, name


# ========== CONTEXT FETCHERS ==========

def fetch_repo_code(repo_url: str, branch: str, token: str) -> str:
    """Load source code files from a GitHub repo using LangChain's GithubFileLoader."""
    repo_id = _parse_repo_id(repo_url)
    try:
        loader = GithubFileLoader(
            repo=repo_id,
            branch=branch,
            file_filter=lambda file_path: file_path.endswith((
                ".txt", ".md", ".html", ".css", ".xml", ".json", ".yaml", ".yml",
                ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".kt", ".kts", ".scala",
                ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".swift", ".m", ".php",
                ".rb", ".pl", ".pm", ".lua", ".sh", ".bash", ".r", ".jl", ".asm",
                ".s", ".dart", ".cs", ".ipynb",
                ".toml", ".cfg", ".ini", ".env.example", ".dockerfile", "Dockerfile",
                "Makefile", ".gradle", ".sbt",
            )),
            access_token=token,
        )
        docs = loader.load()
        full_text = ""
        for i, doc in enumerate(docs, start=1):
            file_name = doc.metadata.get("source", f"file_{i}")
            full_text += f"\n\n===== FILE {i}: {file_name} =====\n"
            full_text += doc.page_content
        logger.info(f"Fetched {len(docs)} code files from {repo_id} ({branch})")
        return full_text
    except Exception as e:
        logger.error(f"Error fetching code from {repo_id}: {e}")
        raise


def fetch_repo_issues(repo_url: str, token: str, state: str = "all", limit: int = 100) -> str:
    """Fetch issues from a GitHub repo."""
    repo_id = _parse_repo_id(repo_url)
    try:
        g = _get_github_client(token)
        repo = g.get_repo(repo_id)
        issues = repo.get_issues(state=state, sort="updated", direction="desc")

        text_parts = []
        count = 0
        for issue in issues:
            if issue.pull_request is not None:
                continue  # Skip PRs (they show up in issues API too)
            if count >= limit:
                break

            labels = ", ".join([l.name for l in issue.labels]) if issue.labels else "none"
            text = f"\n## Issue #{issue.number}: {issue.title}\n"
            text += f"Status: {issue.state} | Labels: {labels} | Created: {issue.created_at}\n"
            if issue.body:
                # Truncate very long issue bodies
                body = issue.body[:2000] + "..." if len(issue.body) > 2000 else issue.body
                text += f"Body: {body}\n"

            # Fetch up to 5 comments per issue
            try:
                comments = issue.get_comments()
                comment_count = 0
                for comment in comments:
                    if comment_count >= 5:
                        break
                    comment_body = comment.body[:500] + "..." if len(comment.body) > 500 else comment.body
                    text += f"  Comment by {comment.user.login}: {comment_body}\n"
                    comment_count += 1
            except Exception:
                pass

            text_parts.append(text)
            count += 1

        result = f"\n\n===== ISSUES ({count} total) =====\n" + "\n".join(text_parts)
        logger.info(f"Fetched {count} issues from {repo_id}")
        return result
    except GithubException as e:
        logger.warning(f"Could not fetch issues from {repo_id}: {e}")
        return f"\n\n===== ISSUES =====\nCould not fetch issues: {e}\n"


def fetch_repo_pull_requests(repo_url: str, token: str, state: str = "all", limit: int = 50) -> str:
    """Fetch pull requests from a GitHub repo."""
    repo_id = _parse_repo_id(repo_url)
    try:
        g = _get_github_client(token)
        repo = g.get_repo(repo_id)
        pulls = repo.get_pulls(state=state, sort="updated", direction="desc")

        text_parts = []
        count = 0
        for pr in pulls:
            if count >= limit:
                break

            text = f"\n## PR #{pr.number}: {pr.title}\n"
            text += f"Status: {pr.state} | Merged: {pr.merged} | Created: {pr.created_at}\n"
            text += f"Base: {pr.base.ref} ← Head: {pr.head.ref}\n"
            if pr.body:
                body = pr.body[:2000] + "..." if len(pr.body) > 2000 else pr.body
                text += f"Body: {body}\n"

            # Get changed files summary
            try:
                files = pr.get_files()
                file_count = 0
                changed_files = []
                for f in files:
                    if file_count >= 20:
                        changed_files.append("... and more files")
                        break
                    changed_files.append(f"  {f.filename} (+{f.additions}/-{f.deletions})")
                    file_count += 1
                if changed_files:
                    text += "Changed files:\n" + "\n".join(changed_files) + "\n"
            except Exception:
                pass

            text_parts.append(text)
            count += 1

        result = f"\n\n===== PULL REQUESTS ({count} total) =====\n" + "\n".join(text_parts)
        logger.info(f"Fetched {count} pull requests from {repo_id}")
        return result
    except GithubException as e:
        logger.warning(f"Could not fetch PRs from {repo_id}: {e}")
        return f"\n\n===== PULL REQUESTS =====\nCould not fetch pull requests: {e}\n"


def fetch_repo_commits(repo_url: str, token: str, branch: str = "main", limit: int = 100) -> str:
    """Fetch recent commit history from a GitHub repo."""
    repo_id = _parse_repo_id(repo_url)
    try:
        g = _get_github_client(token)
        repo = g.get_repo(repo_id)
        commits = repo.get_commits(sha=branch)

        text_parts = []
        count = 0
        for commit in commits:
            if count >= limit:
                break

            author_name = commit.commit.author.name if commit.commit.author else "Unknown"
            date = commit.commit.author.date if commit.commit.author else "Unknown"
            message = commit.commit.message.split("\n")[0]  # First line only
            sha_short = commit.sha[:7]

            stats = ""
            try:
                stats = f" (+{commit.stats.additions}/-{commit.stats.deletions})"
            except Exception:
                pass

            text_parts.append(f"{sha_short} - {message} ({author_name}, {date}){stats}")
            count += 1

        result = f"\n\n===== COMMITS (last {count}) =====\n" + "\n".join(text_parts)
        logger.info(f"Fetched {count} commits from {repo_id} ({branch})")
        return result
    except GithubException as e:
        logger.warning(f"Could not fetch commits from {repo_id}: {e}")
        return f"\n\n===== COMMITS =====\nCould not fetch commits: {e}\n"


def fetch_repo_branches(repo_url: str, token: str) -> str:
    """List all branches from a GitHub repo."""
    repo_id = _parse_repo_id(repo_url)
    try:
        g = _get_github_client(token)
        repo = g.get_repo(repo_id)
        branches = repo.get_branches()

        default_branch = repo.default_branch
        text_parts = []
        count = 0
        for branch in branches:
            marker = " (default)" if branch.name == default_branch else ""
            protected = " [protected]" if branch.protected else ""
            text_parts.append(f"- {branch.name}{marker}{protected}")
            count += 1

        result = f"\n\n===== BRANCHES ({count} total) =====\n" + "\n".join(text_parts)
        logger.info(f"Fetched {count} branches from {repo_id}")
        return result
    except GithubException as e:
        logger.warning(f"Could not fetch branches from {repo_id}: {e}")
        return f"\n\n===== BRANCHES =====\nCould not fetch branches: {e}\n"


def build_full_context(repo_url: str, branch: str, token: str) -> str:
    """
    Orchestrate all fetchers to build a comprehensive repo context.
    Returns one large text blob ready for chunking and vectorization.
    """
    logger.info(f"Building full context for {repo_url} ({branch})")

    sections = []

    # 1. Source code (most important)
    try:
        code_text = fetch_repo_code(repo_url, branch, token)
        sections.append(code_text)
    except Exception as e:
        logger.error(f"Failed to fetch code: {e}")
        sections.append(f"\n\n===== SOURCE CODE =====\nFailed to fetch source code: {e}\n")

    # 2. Issues
    issues_text = fetch_repo_issues(repo_url, token)
    sections.append(issues_text)

    # 3. Pull Requests
    prs_text = fetch_repo_pull_requests(repo_url, token)
    sections.append(prs_text)

    # 4. Commits
    commits_text = fetch_repo_commits(repo_url, token, branch)
    sections.append(commits_text)

    # 5. Branches
    branches_text = fetch_repo_branches(repo_url, token)
    sections.append(branches_text)

    full_context = "\n".join(sections)
    logger.info(f"Full context built: {len(full_context)} characters")
    return full_context


# ========== WEBHOOK HELPERS ==========

def generate_webhook_secret() -> str:
    """Generate a random webhook secret."""
    return secrets.token_hex(32)


def verify_webhook_signature(payload_body: bytes, signature: str, secret: str) -> bool:
    """
    Verify the GitHub webhook HMAC-SHA256 signature.
    The signature header is in format: sha256=<hex_digest>
    """
    if not signature or not signature.startswith("sha256="):
        return False

    expected_sig = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        payload_body,
        hashlib.sha256
    ).hexdigest()

    return hmac.compare_digest(expected_sig, signature)


def register_webhook(repo_url: str, token: str, callback_url: str, secret: str) -> Optional[int]:
    """
    Register a GitHub webhook on the repository.
    Returns the webhook ID if successful, None otherwise.
    """
    repo_id = _parse_repo_id(repo_url)
    try:
        g = _get_github_client(token)
        repo = g.get_repo(repo_id)

        config = {
            "url": callback_url,
            "content_type": "json",
            "secret": secret,
        }
        events = ["push", "issues", "pull_request"]

        hook = repo.create_hook(
            name="web",
            config=config,
            events=events,
            active=True
        )
        logger.info(f"Webhook registered on {repo_id}: hook_id={hook.id}")
        return hook.id
    except GithubException as e:
        logger.error(f"Failed to register webhook on {repo_id}: {e}")
        # Don't fail project creation if webhook registration fails
        # (user might not have admin access to the repo)
        return None


def get_repo_info(repo_url: str, token: str) -> dict:
    """Get basic repo info for display purposes."""
    repo_id = _parse_repo_id(repo_url)
    try:
        g = _get_github_client(token)
        repo = g.get_repo(repo_id)
        return {
            "full_name": repo.full_name,
            "description": repo.description,
            "default_branch": repo.default_branch,
            "stars": repo.stargazers_count,
            "forks": repo.forks_count,
            "language": repo.language,
            "open_issues": repo.open_issues_count,
            "private": repo.private,
        }
    except GithubException as e:
        logger.error(f"Failed to get repo info for {repo_id}: {e}")
        raise
