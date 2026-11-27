const {test} = require('node:test');
const assert = require('node:assert/strict');
const {copyProjectsAfterSync, copyIssueProjects, baseForIssue, issueNumber, doneWhen, managedBody, agentPrompt, syncPullRequest, requestAgent} = require('./learning-workflow.cjs');

const issue = {node_id: 'I_82', number: 82, state: 'open', title: 'Keep vectors aligned', html_url: 'https://github.com/test/repo/issues/82',
  body: '## Why\nUseful.\n\n## Done when\n- [ ] Exact row parity\n- [ ] Offline tests pass\n\n## Notes\n- [ ] Not a criterion'};
const pr = {node_id: 'PR_90', number: 90, body: '', html_url: 'https://github.com/test/repo/pull/90',
  head: {ref: '82-keep-vectors-aligned', sha: 'abc'}, base: {ref: 'week-02-haystack', repo: {full_name: 'test/repo'}}};

function harness() {
  const calls = [];
  const state = {issue, pulls: [], ahead: 1, sha: 'abc', comments: [], assignees: [{login: 'copilot-swe-agent[bot]'}]};
  const endpoint = (name, data) => async args => {
    calls.push([name, args]);
    return {data: typeof data === 'function' ? data() : data};
  };
  const github = {rest: {
    issues: {update: endpoint('milestone', {}), get: endpoint('issue', () => ({...state.issue, assignees: state.assignees})),
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
  assert.ok(updated.includes('\nAuthor notes'));
  assert.ok(updated.endsWith('Closes #82'));
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
  assert.ok(created.body.endsWith('Closes #82'));
  assert.ok(created.body.includes('closes the card explicitly'));
  assert.ok(!created.body.includes('Refs #82'));
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
  assert.ok(updated.endsWith('Closes #82'));
  assert.ok(!updated.includes('Refs #82'));
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


test('closing footer is last for both weekly and default-base PRs', () => {
  for (const options of [{}, {base: 'week-02-haystack', defaultBranch: 'main'}]) {
    const body = managedBody(issue, '', options);
    assert.ok(body.endsWith('\n\nCloses #82'));
    assert.equal(body.match(/^Closes #82$/gm).length, 1);
    assert.ok(body.indexOf('<!-- learning-card:end -->') < body.indexOf('Closes #82'));
  }
});

test('refresh migrates legacy references and preserves author notes and evidence', () => {
  const legacy = 'Author intro\nCloses #82\n\n<!-- learning-card:start -->\n' +
    'Refs #82\n- [x] Exact row parity\n<!-- learning-card:end -->\nAuthor tail\nCloses #82';
  const once = managedBody(issue, legacy, {base: 'week-02-haystack', defaultBranch: 'main'});
  const twice = managedBody(issue, once, {base: 'week-02-haystack', defaultBranch: 'main'});
  assert.equal(once, twice);
  assert.ok(twice.includes('Author intro'));
  assert.ok(twice.includes('Author tail'));
  assert.ok(twice.includes('- [x] Exact row parity'));
  assert.ok(twice.endsWith('Closes #82'));
  assert.equal(twice.match(/^Closes #82$/gm).length, 1);
  assert.ok(!twice.includes('Refs #82'));
});


test('CRLF refresh keeps checked evidence and replaces a tab-padded footer', () => {
  const old = managedBody(issue).replace('- [ ] Exact row parity', '- [x] Exact row parity')
    .replace('Closes #82', 'Closes #82\t ').replaceAll('\n', '\r\n');
  const body = managedBody(issue, old, {base: 'week-02-haystack', defaultBranch: 'main'});
  assert.ok(body.includes('- [x] Exact row parity'));
  assert.ok(body.endsWith('Closes #82'));
  assert.equal(body.match(/^Closes #82[\t ]*\r?$/gm).length, 1);
});


test('a new PR inherits the issue milestone', async () => {
  const h = harness(); h.state.issue = {...issue, milestone: {number: 2}};
  await syncPullRequest(h);
  assert.deepEqual(h.calls.find(([name]) => name === 'milestone')[1], {
    owner: 'test', repo: 'repo', issue_number: 90, milestone: 2
  });
});

test('refresh synchronizes milestone changes and removals without rewriting matching metadata', async () => {
  for (const [source, target, expected] of [[2, 1, 2], [null, 2, null], [2, 2, undefined], [null, null, undefined]]) {
    const h = harness();
    h.state.issue = {...issue, milestone: source === null ? null : {number: source}};
    h.state.pulls = [{...pr, milestone: target === null ? null : {number: target}}];
    await syncPullRequest(h);
    const writes = h.calls.filter(([name]) => name === 'milestone');
    assert.equal(writes.length, expected === undefined ? 0 : 1);
    if (writes.length) assert.equal(writes[0][1].milestone, expected);
  }
});

function projectHarness({source = [], sourceNext = [], failure} = {}) {
  const calls = [];
  const github = {graphql: async (query, variables) => {
    calls.push({query, variables});
    if (failure) throw new Error(failure);
    if (query.startsWith('mutation')) return {addProjectV2ItemById: {item: {id: 'ITEM'}}};
    const next = sourceNext;
    const projects = variables.after ? next : source;
    return {node: {projectItems: {nodes: projects.map(id => ({project: {id}})),
      pageInfo: {hasNextPage: !variables.after && next.length > 0, endCursor: 'next'}}}};
  }};
  return {github, calls};
}

test('project copying adds each source project to the PR without removing unrelated memberships', async () => {
  const h = projectHarness({source: ['P1', 'P2']});
  await copyIssueProjects({...h, issueNodeId: 'I_82', pullNodeId: 'PR_90'});
  assert.deepEqual(h.calls.filter(c => c.query.startsWith('mutation')).map(c => c.variables),
    [{projectId: 'P1', contentId: 'PR_90'}, {projectId: 'P2', contentId: 'PR_90'}]);
  assert.ok(h.calls.every(c => !c.query.includes('deleteProjectV2Item')));
});

test('project copying paginates and deduplicates source project IDs', async () => {
  const h = projectHarness({source: ['P1'], sourceNext: ['P1', 'P2']});
  await copyIssueProjects({...h, issueNodeId: 'I_82', pullNodeId: 'PR_90'});
  assert.equal(h.calls.filter(c => c.variables.after === 'next').length, 1);
  assert.deepEqual(h.calls.filter(c => c.query.startsWith('mutation')).map(c => c.variables.projectId), ['P1', 'P2']);
});

test('no source projects produce no mutations', async () => {
  const h = projectHarness();
  await copyIssueProjects({...h, issueNodeId: 'I_82', pullNodeId: 'PR_90'});
  assert.ok(!h.calls.some(c => c.query.startsWith('mutation')));
});

test('missing project scopes fail explicitly rather than claiming a successful copy', async () => {
  const h = projectHarness({failure: 'INSUFFICIENT_SCOPES'});
  await assert.rejects(copyIssueProjects({...h, issueNodeId: 'I_82', pullNodeId: 'PR_90'}),
    /LEARNING_AGENT_TOKEN.*Projects read\/write.*INSUFFICIENT_SCOPES/);
});

test('inaccessible project membership is not treated as an empty source', async () => {
  const github = {graphql: async () => ({node: null})};
  await assert.rejects(copyIssueProjects({github, issueNodeId: 'I_82', pullNodeId: 'PR_90'}), /Cannot read/);
});


test('successful owner push triggers project copy on the existing current PR', async () => {
  const h = harness(); h.state.pulls = [pr]; h.context.payload.workflow_run.conclusion = 'success';
  const projects = projectHarness({source: ['P1']});
  h.github.graphql = projects.github.graphql;
  await copyProjectsAfterSync(h);
  assert.deepEqual(projects.calls.find(c => c.query.startsWith('mutation')).variables,
    {projectId: 'P1', contentId: 'PR_90'});
});

for (const [name, change] of [
  ['failed PR workflow', h => {h.context.payload.workflow_run.conclusion = 'failure';}],
  ['fork metadata', h => {h.context.payload.workflow_run.head_repository.full_name = 'other/repo';}],
  ['bot metadata', h => {h.context.payload.workflow_run.actor.login = 'bot';}],
  ['stale metadata', h => {h.state.sha = 'new';}],
  ['closed source', h => {h.state.issue = {...issue, state: 'closed'};}],
  ['missing PR', h => {h.state.pulls = [];}],
  ['wrong PR base', h => {h.state.pulls = [{...pr, base: {...pr.base, ref: 'main'}}];}]
]) test(`${name} cannot copy project memberships`, async () => {
  const h = harness(); h.state.pulls = [pr]; h.context.payload.workflow_run.conclusion = 'success';
  change(h);
  h.github.graphql = async () => {throw new Error('Project copy must not run');};
  await copyProjectsAfterSync(h);
});
