const {test} = require('node:test');
const assert = require('node:assert/strict');
const {baseForIssue, issueNumber, doneWhen, managedBody, agentPrompt, syncPullRequest, requestAgent} = require('./learning-workflow.cjs');

const issue = {number: 82, state: 'open', title: 'Keep vectors aligned', html_url: 'https://github.com/test/repo/issues/82',
  body: '## Why\nUseful.\n\n## Done when\n- [ ] Exact row parity\n- [ ] Offline tests pass\n\n## Notes\n- [ ] Not a criterion'};
const pr = {number: 90, body: '', html_url: 'https://github.com/test/repo/pull/90',
  head: {ref: '82-keep-vectors-aligned', sha: 'abc'}, base: {ref: 'week-02-haystack', repo: {full_name: 'test/repo'}}};

function harness() {
  const calls = [];
  const state = {issue, pulls: [], ahead: 1, sha: 'abc', comments: [], assignees: [{login: 'copilot-swe-agent[bot]'}]};
  const endpoint = (name, data) => async args => {
    calls.push([name, args]);
    return {data: typeof data === 'function' ? data() : data};
  };
  const github = {rest: {
    issues: {get: endpoint('issue', () => ({...state.issue, assignees: state.assignees})),
      addAssignees: endpoint('assign', {}), createComment: endpoint('comment', {}), listComments: 'comments'},
    repos: {get: endpoint('repo', {default_branch: 'main'}),
      compareCommitsWithBasehead: endpoint('compare', () => ({ahead_by: state.ahead})),
      getBranch: endpoint('branch', () => ({commit: {sha: state.sha}}))},
    pulls: {list: endpoint('pulls', () => state.pulls), create: endpoint('create', pr), update: endpoint('update', pr)},
    actions: {createWorkflowDispatch: endpoint('dispatch', {})}
  }, paginate: async () => state.comments};
  const context = {repo: {owner: 'test', repo: 'repo'}, payload: {
    ref: 'refs/heads/82-keep-vectors-aligned', sender: {type: 'User', login: 'test'},
    workflow_run: {head_branch: pr.head.ref, head_sha: 'abc', actor: {login: 'test'}, head_repository: {full_name: 'test/repo'}}
  }};
  return {github, context, core: {info() {}}, state, calls};
}

test('only leading GitHub issue-number branch names are accepted', () => {
  assert.equal(issueNumber('82-w02-d2-adapter'), 82);
  for (const name of ['main', 'week-01-own-implementation', 'copilot/82-tests', 'fix/82-bug', '0-bad']) {
    assert.ok(!issueNumber(name));
  }
});

test('extract Done when only, with CRLF support and no next-section leakage', () => {
  assert.deepEqual(doneWhen(issue.body.replaceAll('\n', '\r\n')), ['Exact row parity', 'Offline tests pass']);
  assert.deepEqual(doneWhen('No checklist'), []);
});

test('refresh preserves author text and matching checked criteria without checking new ones', () => {
  const first = managedBody(issue).replace('- [ ] Exact row parity', '- [x] Exact row parity');
  const old = 'Author introduction\n' + first + '\nAuthor notes';
  const updated = managedBody({...issue, body: issue.body.replace('Offline tests pass', 'New evidence')}, old);
  assert.ok(updated.startsWith('Author introduction\n'));
  assert.ok(updated.endsWith('\nAuthor notes'));
  assert.ok(updated.includes('- [x] Exact row parity'));
  assert.ok(updated.includes('- [ ] New evidence'));
  assert.ok(!updated.includes('Offline tests pass'));
  assert.equal(updated.split('learning-card:start').length, 2);
});

test('branch creation at the base tip does not open an empty PR', async () => {
  const h = harness(); h.state.ahead = 0;
  await syncPullRequest(h);
  assert.ok(!h.calls.some(([name]) => name === 'create'));
});

test('first real owner push opens a draft, links the issue and explicitly dispatches CI', async () => {
  const h = harness(); await syncPullRequest(h);
  const created = h.calls.find(([name]) => name === 'create')[1];
  assert.equal(created.draft, true);
  assert.equal(created.base, 'week-02-haystack');
  assert.equal(h.calls.find(([name]) => name === 'compare')[1].basehead, 'week-02-haystack...82-keep-vectors-aligned');
  assert.ok(created.body.includes('Refs #82'));
  assert.ok(created.body.includes('closes the card explicitly'));
  assert.ok(!created.body.includes('Closes #82'));
  assert.ok(created.body.includes('- [ ] Exact row parity'));
  assert.equal(h.calls.find(([name]) => name === 'dispatch')[1].ref, pr.head.ref);
});

test('subsequent pushes refresh the existing PR, never duplicate or dispatch extra CI', async () => {
  const h = harness(); h.state.pulls = [pr];
  await syncPullRequest(h);
  assert.equal(h.calls.filter(([name]) => name === 'update').length, 1);
  assert.ok(!h.calls.some(([name]) => ['create', 'dispatch'].includes(name)));
});

for (const [name, change] of [
  ['bot push', h => {h.context.payload.sender = {type: 'Bot', login: 'copilot-swe-agent[bot]'};}],
  ['other user', h => {h.context.payload.sender.login = 'someone';}],
  ['deleted branch', h => {h.context.payload.deleted = true;}],
  ['archive branch', h => {h.context.payload.ref = 'refs/heads/week-01-own-implementation';}]
]) test(`${name} never opens a PR`, async () => {
  const h = harness(); change(h); await syncPullRequest(h); assert.deepEqual(h.calls, []);
});

test('closed issue and PR-number branch are not interpreted as open learning cards', async () => {
  for (const extra of [{state: 'closed'}, {pull_request: {}}]) {
    const h = harness(); h.state.issue = {...issue, ...extra}; await syncPullRequest(h);
    assert.ok(!h.calls.some(([name]) => name === 'create'));
  }
});

test('agent request targets the current existing PR and explicitly prohibits implementation', async () => {
  const h = harness(); h.state.pulls = [pr]; await requestAgent(h);
  const comment = h.calls.find(([name]) => name === 'comment')[1];
  assert.equal(comment.issue_number, pr.number);
  assert.ok(comment.body.startsWith('@copilot'));
  assert.ok(comment.body.includes('do not implement'));
  assert.ok(comment.body.includes('learning-agent:abc'));
});

for (const [name, change] of [
  ['stale head', h => {h.state.sha = 'newer';}],
  ['agent-originated CI', h => {h.context.payload.workflow_run.actor.login = 'copilot-swe-agent[bot]';}],
  ['fork', h => {h.context.payload.workflow_run.head_repository.full_name = 'other/repo';}],
  ['different PR head', h => {h.state.pulls = [{...pr, head: {...pr.head, sha: 'newer'}}];}],
  ['already requested commit', h => {h.state.comments = [{user: {login: 'test'}, body: '<!-- learning-agent:abc -->'}];}]
]) test(`${name} never starts duplicate or unsafe agent work`, async () => {
  const h = harness(); h.state.pulls = [pr]; change(h); await requestAgent(h);
  assert.ok(!h.calls.some(([name]) => ['assign', 'comment'].includes(name)));
});

test('failed Copilot assignment is explicit, not a successful no-op', async () => {
  const h = harness(); h.state.pulls = [pr]; h.state.assignees = [];
  await assert.rejects(requestAgent(h), /Copilot assignment was not accepted/);
  assert.ok(!h.calls.some(([name]) => name === 'comment'));
});

test('prompt requires evidence and preserves human review', () => {
  const text = agentPrompt({issue, pr, sha: 'abc'});
  for (const clause of ['Never merge', 'Uncheck unsupported items', 'offline tests', 'learning-tests.agent.md']) {
    assert.ok(text.includes(clause));
  }
});


test('a manually opened PR with a null body receives the issue checklist', () => {
  const body = managedBody(issue, null);
  assert.ok(body.includes('Closes #82'));
  assert.ok(body.includes('- [ ] Exact row parity'));
});


test('only explicitly prepared Week 2 cards use the weekly base', () => {
  for (const number of [22, 23, 24, 25, 26, 82, 83, 99]) {
    assert.equal(baseForIssue(number, 'main'), 'week-02-haystack');
  }
  for (const number of [1, 21, 27, 81, 100]) {
    assert.equal(baseForIssue(number, 'main'), 'main');
  }
});

test('non-week-2 cards retain the default base and closing keyword', async () => {
  const h = harness(); h.state.issue = {...issue, number: 7};
  h.context.payload.ref = 'refs/heads/7-other-task';
  await syncPullRequest(h);
  const created = h.calls.find(([name]) => name === 'create')[1];
  assert.equal(created.base, 'main');
  assert.ok(created.body.includes('Closes #7'));
});

test('weekly refresh preserves checked evidence and author text', () => {
  const options = {base: 'week-02-haystack', defaultBranch: 'main'};
  const old = 'Author notes\n' + managedBody(issue).replace('- [ ] Exact row parity', '- [x] Exact row parity');
  const updated = managedBody(issue, old, options);
  assert.ok(updated.startsWith('Author notes'));
  assert.ok(updated.includes('- [x] Exact row parity'));
  assert.ok(updated.includes('Refs #82'));
  assert.ok(!updated.includes('Closes #82'));
});

test('a weekly PR aimed at main cannot dispatch the test agent', async () => {
  const h = harness(); h.state.pulls = [{...pr, base: {...pr.base, ref: 'main'}}];
  await requestAgent(h);
  assert.ok(!h.calls.some(([name]) => ['assign', 'comment'].includes(name)));
});


test('an existing wrong-base PR is retargeted rather than duplicated', async () => {
  const h = harness(); h.state.pulls = [{...pr, base: {...pr.base, ref: 'main'}}];
  await syncPullRequest(h);
  assert.equal(h.calls.find(([name]) => name === 'update')[1].base, 'week-02-haystack');
  assert.ok(!h.calls.some(([name]) => name === 'create'));
});
