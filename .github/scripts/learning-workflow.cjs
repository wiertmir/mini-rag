const START = '<!-- learning-card:start -->';
const END = '<!-- learning-card:end -->';

function issueNumber(branch) {
  const match = /^(\d+)-/.exec(branch);
  return match ? Number(match[1]) : null;
}

function doneWhen(body) {
  const section = /^## Done when\s*\r?\n([\s\S]*?)(?=^## |^---\s*$|(?![\s\S]))/im.exec(body || '');
  return (section?.[1] || '').split(/\r?\n/)
    .filter(line => /^- \[[ xX]\] /.test(line))
    .map(line => line.replace(/^- \[[ xX]\] /, ''));
}

function managedBody(issue, oldBody = '') {
  oldBody = oldBody || '';
  const completed = new Set(
    (oldBody.split(START)[1]?.split(END)[0] || '').split('\n')
      .filter(line => /^- \[[xX]\] /.test(line)).map(line => line.slice(6))
  );
  const criteria = doneWhen(issue.body);
  const block = `${START}\n## Learning card\n\nCloses #${issue.number}\n\n` +
    `Source: ${issue.html_url}\n\n## Done when (evidence required)\n\n` +
    (criteria.length ? criteria.map(text => `- [${completed.has(text) ? 'x' : ' '}] ${text}`).join('\n') :
      '- [ ] Review the issue contract and define measurable completion criteria.') +
    `\n\nA push is not proof of completion. The test agent checks criteria only with code/test/artifact evidence.\n${END}`;
  if (oldBody.includes(START) && oldBody.includes(END)) {
    const begin = oldBody.indexOf(START), end = oldBody.indexOf(END) + END.length;
    return oldBody.slice(0, begin) + block + oldBody.slice(end);
  }
  return `${oldBody || '## What changed\n\nImplementation of the linked learning card; see the diff and checklist.'}\n\n${block}`;
}

async function syncPullRequest({github, context, core}) {
  const {owner, repo} = context.repo;
  const event = context.payload;
  const branch = event.ref?.replace(/^refs\/heads\//, '');
  const number = issueNumber(branch || '');
  if (!number || event.deleted || event.sender?.type !== 'User' || event.sender.login !== owner) return;
  const {data: issue} = await github.rest.issues.get({owner, repo, issue_number: number});
  if (issue.pull_request || issue.state !== 'open') return;
  const {data: metadata} = await github.rest.repos.get({owner, repo});
  const base = metadata.default_branch;
  const {data: comparison} = await github.rest.repos.compareCommitsWithBasehead({
    owner, repo, basehead: `${base}...${branch}`
  });
  // GitHub's Create branch starts at the base tip: wait for an actual commit.
  if (!comparison.ahead_by) return;
  const {data: pulls} = await github.rest.pulls.list({owner, repo, state: 'open', head: `${owner}:${branch}`, base});
  let pr = pulls[0];
  if (pr) {
    await github.rest.pulls.update({owner, repo, pull_number: pr.number, body: managedBody(issue, pr.body)});
  } else {
    ({data: pr} = await github.rest.pulls.create({owner, repo, base, head: branch,
      draft: true, title: `${issue.title} (#${number})`, body: managedBody(issue)}));
    // GITHUB_TOKEN-created PRs do not trigger pull_request workflows.
    await github.rest.actions.createWorkflowDispatch({owner, repo, workflow_id: 'ci.yml', ref: branch});
  }
  core.info(`Learning PR: ${pr.html_url}`);
}

function agentPrompt({issue, pr, sha}) {
  return `@copilot Follow .github/agents/learning-tests.agent.md for this existing PR.\n\n` +
    `Assess learner commit ${sha} against issue #${issue.number}: ${issue.html_url}.\n` +
    `Work only on this PR's branch (${pr.head.ref}); do not open a separate implementation PR.\n` +
    `Read the issue's contract and Done when checklist. Add or update meaningful offline tests for the changed behavior, ` +
    `only when useful; do not implement the learner's production solution or weaken existing tests. ` +
    `For documentation-only changes, explain why no tests are needed.\n` +
    `Run focused checks. Update the managed checklist in this PR body with code, test names and artifact evidence. ` +
    `Uncheck unsupported items; a commit or green CI alone does not prove every criterion. ` +
    `Do not check live/milestone/PDF requirements you have not verified. ` +
    `Leave a concise evidence report, including failures and unverified items. Never merge or close the issue.\n` +
    `If the head changed while you worked, reassess the new head before pushing; do not force-push.\n\n` +
    `<!-- learning-agent:${sha} -->`;
}

async function requestAgent({github, context, core}) {
  const {owner, repo} = context.repo;
  const run = context.payload.workflow_run;
  const number = issueNumber(run.head_branch || '');
  if (!number || run.head_repository?.full_name !== `${owner}/${repo}` || run.actor?.login !== owner) return;
  const {data: branch} = await github.rest.repos.getBranch({owner, repo, branch: run.head_branch});
  if (branch.commit.sha !== run.head_sha) return; // superseded CI must not dispatch stale work
  const {data: pulls} = await github.rest.pulls.list({owner, repo, state: 'open', head: `${owner}:${run.head_branch}`});
  const pr = pulls[0];
  if (!pr || pr.head.sha !== run.head_sha || pr.base.repo.full_name !== `${owner}/${repo}`) return;
  const {data: issue} = await github.rest.issues.get({owner, repo, issue_number: number});
  if (issue.pull_request || issue.state !== 'open') return;
  const comments = await github.paginate(github.rest.issues.listComments, {owner, repo, issue_number: pr.number});
  if (comments.some(c => c.user?.login === owner && c.body?.includes(`<!-- learning-agent:${run.head_sha} -->`))) return;
  await github.rest.issues.addAssignees({owner, repo, issue_number: pr.number, assignees: ['copilot-swe-agent[bot]']});
  const {data: assigned} = await github.rest.issues.get({owner, repo, issue_number: pr.number});
  if (!assigned.assignees.some(a => a.login === 'copilot-swe-agent[bot]')) {
    throw new Error('Copilot assignment was not accepted; check agent availability and user-token permissions.');
  }
  await github.rest.issues.createComment({owner, repo, issue_number: pr.number,
    body: agentPrompt({issue, pr, sha: run.head_sha})});
  core.info(`Requested Copilot tests and checklist assessment for ${pr.html_url} at ${run.head_sha}`);
}

module.exports = {issueNumber, doneWhen, managedBody, agentPrompt, syncPullRequest, requestAgent};
