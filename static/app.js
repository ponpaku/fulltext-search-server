/**
 * フォルダ内テキスト検索 — YomiToku Style
 * System Version: 1.2.0
 */

const state = {
  folders: [],
  selected: new Set(),
  mode: 'AND',
  keywords: [],
  viewMode: 'hit',
  results: [],
  filteredResults: null,
  folderListOpen: true,
  isSearching: false,
  spaceMode: 'jp',
  renderedCount: 0,
  groupedResults: null,
  isRenderingBatch: false,
  normalizeMode: 'normalized',
  fileModalFolderId: null,
  fileModalFolderName: '',
  fileModalScope: 'indexed',
  queryHistory: [],
  historyModalOpen: false,
  currentIndexUuid: null,
  detailCache: new Map(),
  detailRequests: new Map(),
  filter: {
    folders: new Set(),
    extensions: new Set(),
  },
  filterOptions: {
    folders: [],
    extensions: [],
  },
  clientId: null,
};

// DOM Elements
const $ = (id) => document.getElementById(id);
const escapeHtml = (value) => (
  String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
);
const folderListEl = $('folderList');
const folderStatusEl = $('folderStatus');
const resultsEl = $('results');
const resultCountEl = $('resultCount');
const modeGroup = $('modeGroup');
const rangeInput = $('range');
const spaceModeSelect = $('spaceMode');
const normalizeModeSelect = $('normalizeMode');
const searchForm = $('searchForm');
const queryInput = $('query');
const viewToggle = $('viewToggle');
const copyPathsBtn = $('copyPaths');
const folderToggle = $('folderToggle');
const refreshBtn = $('refreshFolders');
const themeToggle = $('themeToggle');
const helpToggle = $('helpToggle');
const statusChips = $('statusChips');
const fileModal = $('fileModal');
const fileModalClose = $('fileModalClose');
const fileModalTitle = $('fileModalTitle');
const fileListContent = $('fileListContent');
const fileScopeToggle = $('fileScopeToggle');
const helpModal = $('helpModal');
const helpModalClose = $('helpModalClose');
const historyToggle = $('historyToggle');
const historyModal = $('historyModal');
const historyModalClose = $('historyModalClose');
const historyListContent = $('historyListContent');
const clearHistoryBtn = $('clearHistoryBtn');
const exportBtn = $('exportBtn');
const filterBtn = $('filterBtn');
const noticeBar = $('noticeBar');
const filterPanel = $('filterPanel');
const filterFoldersEl = $('filterFolders');
const filterExtensionsEl = $('filterExtensions');
const applyFilterBtn = $('applyFilter');
const clearFilterBtn = $('clearFilter');
const closeFilterBtn = $('closeFilter');

let lastNormalizeNotice = null;

const showNotice = (message) => {
  if (!noticeBar) return;
  noticeBar.textContent = message;
  noticeBar.style.display = 'flex';
  clearTimeout(showNotice._timer);
  showNotice._timer = setTimeout(() => {
    noticeBar.style.display = 'none';
  }, 4000);
};

const getNormalizeLabel = (mode) => ({
  exact: '厳格（最小整形）',
  normalized: 'ゆらぎ吸収',
}[mode] || mode);

// ═══════════════════════════════════════════════════════════════
// HEARTBEAT
// ═══════════════════════════════════════════════════════════════

const HEARTBEAT_CLIENT_KEY = 'fts_client_id';
const HEARTBEAT_INTERVAL_MS = 35000;
const HEARTBEAT_JITTER_MS = 10000;
const HEARTBEAT_MIN_GAP_MS = 5000;
const HEARTBEAT_INTERACTION_GAP_MS = 15000;
const HEARTBEAT_IDLE_THRESHOLD_MS = 90000;

let heartbeatTimer = null;
let heartbeatInFlight = false;
let lastHeartbeatAt = 0;
let lastInteractionHeartbeatAt = 0;
let lastUserActivityAt = 0;

const getClientId = () => {
  let clientId = localStorage.getItem(HEARTBEAT_CLIENT_KEY);
  if (!clientId) {
    clientId = window.crypto?.randomUUID?.()
      || `${Date.now()}-${Math.random().toString(36).slice(2, 12)}`;
    localStorage.setItem(HEARTBEAT_CLIENT_KEY, clientId);
  }
  return clientId;
};

const sendHeartbeat = async (reason = 'interval', force = false) => {
  if (heartbeatInFlight) return;
  const now = Date.now();
  if (!force && now - lastHeartbeatAt < HEARTBEAT_MIN_GAP_MS) return;
  if (!force && lastUserActivityAt && now - lastUserActivityAt > HEARTBEAT_IDLE_THRESHOLD_MS) return;
  heartbeatInFlight = true;
  try {
    await fetch('/api/heartbeat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ client_id: state.clientId }),
    });
  } catch (err) {
    console.warn('heartbeat failed', reason, err);
  } finally {
    lastHeartbeatAt = now;
    heartbeatInFlight = false;
  }
};

const scheduleHeartbeat = () => {
  const jitter = Math.floor(Math.random() * HEARTBEAT_JITTER_MS);
  clearTimeout(heartbeatTimer);
  heartbeatTimer = setTimeout(async () => {
    await sendHeartbeat('interval');
    scheduleHeartbeat();
  }, HEARTBEAT_INTERVAL_MS + jitter);
};

const handleHeartbeatInteraction = () => {
  const now = Date.now();
  lastUserActivityAt = now;
  if (now - lastInteractionHeartbeatAt < HEARTBEAT_INTERACTION_GAP_MS) return;
  lastInteractionHeartbeatAt = now;
  sendHeartbeat('interaction', true);
};

const initHeartbeat = () => {
  state.clientId = getClientId();
  lastUserActivityAt = Date.now();
  sendHeartbeat('init', true);
  scheduleHeartbeat();
  window.addEventListener('focus', () => {
    lastUserActivityAt = Date.now();
    sendHeartbeat('focus', true);
  });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      lastUserActivityAt = Date.now();
      sendHeartbeat('visible', true);
    }
  });
  ['click', 'keydown', 'pointerdown', 'touchstart'].forEach((eventName) => {
    document.addEventListener(eventName, handleHeartbeatInteraction);
  });
};

// ═══════════════════════════════════════════════════════════════
// CLIPBOARD PERMISSION
// ═══════════════════════════════════════════════════════════════

const requestClipboardPermission = async () => {
  if (!navigator.permissions?.query) return true;
  try {
    const status = await navigator.permissions.query({ name: 'clipboard-write' });
    if (status.state === 'denied') {
      alert('クリップボード書き込みがブロックされています。ブラウザ設定で許可してください。');
      return false;
    }
    if (status.state === 'prompt') {
      const proceed = confirm('クリップボードへのコピー許可をブラウザに求めます。よろしいですか？');
      if (!proceed) return false;
    }
    return true;
  } catch (err) {
    console.warn('clipboard permission check failed', err);
    return true;
  }
};

const copyTextToClipboard = async (text) => {
  const allowed = await requestClipboardPermission();
  if (!allowed) return false;

  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (err) {
      console.warn('navigator.clipboard failed, fallback to execCommand', err);
    }
  }

  try {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.top = '-9999px';
    textarea.setAttribute('readonly', '');
    document.body.appendChild(textarea);
    textarea.select();
    const success = document.execCommand('copy');
    document.body.removeChild(textarea);
    if (!success) throw new Error('execCommand copy failed');
    return true;
  } catch (err) {
    console.warn('fallback copy failed', err);
    alert('コピーに失敗しました');
    return false;
  }
};

// ═══════════════════════════════════════════════════════════════
// THEME
// ═══════════════════════════════════════════════════════════════

const initTheme = () => {
  const saved = localStorage.getItem('theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  const theme = saved || (prefersDark ? 'dark' : 'light');
  document.documentElement.dataset.theme = theme;
};

const toggleTheme = () => {
  const current = document.documentElement.dataset.theme;
  const next = current === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  localStorage.setItem('theme', next);
};

// ═══════════════════════════════════════════════════════════════
// QUERY HISTORY
// ═══════════════════════════════════════════════════════════════

const HISTORY_STORAGE_KEY = 'searchQueryHistory';
const HISTORY_MAX_ITEMS = 30;

const loadQueryHistory = () => {
  try {
    const stored = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (stored) {
      state.queryHistory = JSON.parse(stored);
    }
  } catch (err) {
    console.warn('Failed to load query history', err);
    state.queryHistory = [];
  }
};

const saveQueryHistory = () => {
  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(state.queryHistory));
  } catch (err) {
    console.warn('Failed to save query history', err);
  }
};

const addToQueryHistory = (query, mode, range, spaceMode, normalizeMode, folders, resultCount, indexUuid) => {
  const timestamp = Date.now();
  const historyItem = {
    id: `${timestamp}-${Math.random().toString(36).substr(2, 9)}`,
    query,
    mode,
    range_limit: range,
    space_mode: spaceMode,
    normalize_mode: normalizeMode,
    folders: [...folders],
    result_count: resultCount,
    index_uuid: indexUuid,
    timestamp,
    pinned: false,
  };

  // Remove duplicate (same query and params)
  state.queryHistory = state.queryHistory.filter(item => {
    if (item.pinned) return true; // Keep pinned items
    return !(
      item.query === query &&
      item.mode === mode &&
      item.range_limit === range &&
      item.space_mode === spaceMode &&
      (item.normalize_mode || 'exact') === normalizeMode &&
      JSON.stringify(item.folders.sort()) === JSON.stringify([...folders].sort())
    );
  });

  // Add to beginning
  state.queryHistory.unshift(historyItem);

  // Keep only max items (excluding pinned)
  const pinned = state.queryHistory.filter(item => item.pinned);
  const unpinned = state.queryHistory.filter(item => !item.pinned).slice(0, HISTORY_MAX_ITEMS);
  state.queryHistory = [...pinned, ...unpinned];

  saveQueryHistory();
};

const togglePinHistory = (itemId) => {
  const item = state.queryHistory.find(h => h.id === itemId);
  if (item) {
    item.pinned = !item.pinned;
    saveQueryHistory();
  }
};

const deleteHistoryItem = (itemId) => {
  state.queryHistory = state.queryHistory.filter(h => h.id !== itemId);
  saveQueryHistory();
};

const clearUnpinnedHistory = () => {
  state.queryHistory = state.queryHistory.filter(h => h.pinned);
  saveQueryHistory();
};

// ═══════════════════════════════════════════════════════════════
// STATUS CHIPS
// ═══════════════════════════════════════════════════════════════

const updateStatusChips = (folders) => {
  const total = folders.length;
  const ready = folders.filter(f => f.ready).length;
  
  if (total === 0) {
    statusChips.innerHTML = '<span class="chip error">未設定</span>';
  } else if (ready === total) {
    statusChips.innerHTML = `
      <span class="chip positive">${ready}/${total} 準備完了</span>
      <span class="chip">検索可能</span>
    `;
  } else {
    statusChips.innerHTML = `
      <span class="chip warning">${ready}/${total} 準備中</span>
    `;
  }
};

// ═══════════════════════════════════════════════════════════════
// FOLDER STATUS
// ═══════════════════════════════════════════════════════════════

const renderFolderStatus = (folders) => {
  if (!folders.length) {
    folderStatusEl.innerHTML = `
      <div class="status-item error">
        <span class="status-dot"></span>
        <div class="status-info">
          <span class="status-name">未設定</span>
          <span class="status-detail">.env で SEARCH_FOLDERS を設定してください</span>
        </div>
      </div>
    `;
    return;
  }

  folderStatusEl.innerHTML = folders.map(f => {
    const statusClass = f.ready ? 'ready' : (f.message ? 'error' : '');
    const indexed = f.stats?.indexed_files 
      ? `${f.stats.indexed_files}/${f.stats.total_files ?? '-'} 件` 
      : '';
    const safeName = escapeHtml(f.name);
    const safeMessage = f.message ? escapeHtml(f.message) : '';
    
    return `
      <div class="status-item ${statusClass}" data-id="${f.id}">
        <span class="status-dot"></span>
        <div class="status-info">
          <span class="status-name">${safeName}</span>
          <span class="status-detail">${f.ready ? '検索可' : '処理中'} ${indexed} ${safeMessage}</span>
        </div>
      </div>
    `;
  }).join('');
};

// ═══════════════════════════════════════════════════════════════
// FOLDER LIST
// ═══════════════════════════════════════════════════════════════

const renderFolderList = (folders) => {
  if (!folders.length) {
    folderListEl.innerHTML = '<div class="loading-placeholder">フォルダが設定されていません</div>';
    return;
  }

  folderListEl.innerHTML = folders.map(f => {
    const checked = f.ready ? 'checked' : '';
    const disabled = f.ready ? '' : 'disabled';
    const fileCount = f.stats?.total_files ? `${f.stats.total_files} ファイル` : '';
    const safeName = escapeHtml(f.name);
    const safePath = escapeHtml(f.displayPath || f.path);
    
    return `
      <label class="folder-item ${f.ready && state.selected.has(f.id) ? 'selected' : ''}">
        <input type="checkbox" data-id="${f.id}" ${checked} ${disabled}>
        <div class="folder-info">
          <div class="folder-name">${safeName}</div>
          <div class="folder-path">${safePath}</div>
          <div class="folder-meta">
            <span class="chip ${f.ready ? 'positive' : ''}">${f.ready ? '検索可' : '準備中'}</span>
            ${fileCount ? `<span class="chip">${fileCount}</span>` : ''}
          </div>
        </div>
      </label>
    `;
  }).join('');

  // Event listeners
  folderListEl.querySelectorAll('input[type="checkbox"]').forEach(cb => {
    const folderId = cb.dataset.id;
    if (cb.checked) state.selected.add(folderId);
    
    cb.addEventListener('change', () => {
      if (cb.checked) {
        state.selected.add(folderId);
        cb.closest('.folder-item').classList.add('selected');
      } else {
        state.selected.delete(folderId);
        cb.closest('.folder-item').classList.remove('selected');
      }
    });
  });

  folderListEl.classList.toggle('open', state.folderListOpen);
};

// ═══════════════════════════════════════════════════════════════
// LOAD FOLDERS
// ═══════════════════════════════════════════════════════════════

const loadFolders = async () => {
  folderStatusEl.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';
  folderListEl.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';
  refreshBtn.disabled = true;

  try {
    const res = await fetch('/api/folders');
    const data = await res.json();
    state.folders = data.folders || [];
    state.selected = new Set(state.folders.filter(f => f.ready).map(f => f.id));
    
    updateStatusChips(state.folders);
    renderFolderStatus(state.folders);
    renderFolderList(state.folders);
  } catch (err) {
    folderStatusEl.innerHTML = `
      <div class="status-item error">
        <span class="status-dot"></span>
        <div class="status-info">
          <span class="status-name">エラー</span>
          <span class="status-detail">フォルダ情報の取得に失敗しました</span>
        </div>
      </div>
    `;
    statusChips.innerHTML = '<span class="chip error">エラー</span>';
    console.error(err);
  } finally {
    refreshBtn.disabled = false;
  }
};

// ═══════════════════════════════════════════════════════════════
// MODE SWITCHING
// ═══════════════════════════════════════════════════════════════

const setMode = (mode) => {
  state.mode = mode;
  modeGroup.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.mode === mode);
  });
  rangeInput.disabled = mode === 'OR';
  if (mode === 'OR') rangeInput.value = 0;
};

// ═══════════════════════════════════════════════════════════════
// RESULT VIEW SWITCHING
// ═══════════════════════════════════════════════════════════════

const setViewMode = (mode) => {
  state.viewMode = mode;
  viewToggle.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.view === mode);
  });
  renderResults();
};

const updateFolderToggleLabel = () => {
  if (!folderToggle) return;
  folderToggle.textContent = state.folderListOpen ? '閉じる' : '開く';
};

const toggleFolderList = () => {
  state.folderListOpen = !state.folderListOpen;
  folderListEl.classList.toggle('open', state.folderListOpen);
  updateFolderToggleLabel();
};

// ═══════════════════════════════════════════════════════════════
// FILE LIST MODAL
// ═══════════════════════════════════════════════════════════════

const closeFileModal = () => {
  fileModal.classList.remove('open');
  fileModal.setAttribute('aria-hidden', 'true');
};

const setFileScope = (scope) => {
  if (!fileScopeToggle) return;
  fileScopeToggle.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.scope === scope);
  });
};

const loadFileList = async (folderId, folderName, scope) => {
  fileModalTitle.textContent = `${folderName} のファイル一覧`;
  fileListContent.innerHTML = '<div class="loading-placeholder">読み込み中...</div>';

  try {
    const res = await fetch(`/api/folders/${folderId}/files?scope=${scope}`);
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || '取得に失敗しました');
    }
    const data = await res.json();
    if (!data.files?.length) {
      fileListContent.innerHTML = '<div class="loading-placeholder">ファイルがありません</div>';
      return;
    }

    const items = data.files.map(f => {
      const depth = f.relative.replace(/\\\\/g, '/').replace(/\\/g, '/').split('/').length - 1;
      const safeRel = escapeHtml(f.relative);
      const reason = f.reason ? `<span class="file-reason">理由: ${escapeHtml(f.reason)}</span>` : '';
      const statusLabel = data.scope === 'all'
        ? `<span class="file-status ${f.indexed ? 'indexed' : 'missing'}">${f.indexed ? '済' : '未'}</span>`
        : '';
      return `
        <div class="file-item" style="--depth:${depth}">
          <div class="file-meta">
            <span class="file-name">${safeRel}</span>
            ${reason}
          </div>
          ${statusLabel}
        </div>
      `;
    }).join('');
    fileListContent.innerHTML = `
      <div class="file-list">
        ${items}
      </div>
    `;
  } catch (err) {
    fileListContent.innerHTML = `<div class="loading-placeholder">エラー: ${escapeHtml(err.message)}</div>`;
    console.error(err);
  }
};

const openFileModal = async (folderId, folderName) => {
  fileModal.classList.add('open');
  fileModal.setAttribute('aria-hidden', 'false');
  state.fileModalFolderId = folderId;
  state.fileModalFolderName = folderName;
  state.fileModalScope = state.fileModalScope || 'indexed';
  setFileScope(state.fileModalScope);
  await loadFileList(folderId, folderName, state.fileModalScope);
};

if (fileScopeToggle) {
  fileScopeToggle.addEventListener('click', (e) => {
    const btn = e.target.closest('.toggle-btn');
    if (!btn) return;
    const scope = btn.dataset.scope;
    state.fileModalScope = scope;
    setFileScope(scope);
    if (state.fileModalFolderId) {
      loadFileList(state.fileModalFolderId, state.fileModalFolderName, scope);
    }
  });
}

fileModal.addEventListener('click', (e) => {
  if (e.target === fileModal) closeFileModal();
});
fileModalClose.addEventListener('click', closeFileModal);

// ═══════════════════════════════════════════════════════════════
// HELP MODAL
// ═══════════════════════════════════════════════════════════════

const closeHelpModal = () => {
  helpToggle?.focus();
  helpModal.classList.remove('open');
  helpModal.setAttribute('aria-hidden', 'true');
};

const openHelpModal = () => {
  helpToggle?.blur();
  helpModal.classList.add('open');
  helpModal.setAttribute('aria-hidden', 'false');
};

helpToggle.addEventListener('click', openHelpModal);
helpModal.addEventListener('click', (e) => {
  if (e.target === helpModal) closeHelpModal();
});
helpModalClose.addEventListener('click', closeHelpModal);

// ═══════════════════════════════════════════════════════════════
// HISTORY MODAL
// ═══════════════════════════════════════════════════════════════

const closeHistoryModal = () => {
  historyToggle?.focus();
  historyModal.classList.remove('open');
  historyModal.setAttribute('aria-hidden', 'true');
};

const openHistoryModal = () => {
  historyToggle?.blur();
  historyModal.classList.add('open');
  historyModal.setAttribute('aria-hidden', 'false');
  renderHistoryList();
};

const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp);
  const now = Date.now();
  const diff = now - timestamp;
  const minutes = Math.floor(diff / 60000);
  const hours = Math.floor(diff / 3600000);
  const days = Math.floor(diff / 86400000);

  if (minutes < 1) return '今';
  if (minutes < 60) return `${minutes}分前`;
  if (hours < 24) return `${hours}時間前`;
  if (days < 7) return `${days}日前`;

  return date.toLocaleDateString('ja-JP', { month: 'short', day: 'numeric' });
};

const executeHistorySearch = async (item) => {
  closeHistoryModal();

  // Restore form state
  queryInput.value = item.query;
  setMode(item.mode);
  rangeInput.value = item.range_limit || 0;
  spaceModeSelect.value = item.space_mode || 'jp';
  normalizeModeSelect.value = item.normalize_mode || 'exact';

  // Restore folder selection
  state.selected = new Set(item.folders);
  renderFolderList(state.folders);

  // Execute search
  await runSearch();
};

const renderHistoryList = () => {
  if (!state.queryHistory || state.queryHistory.length === 0) {
    historyListContent.innerHTML = '<div class="loading-placeholder">履歴がありません</div>';
    return;
  }

  const items = state.queryHistory.map(item => {
    const folderNames = item.folders
      .map(fid => state.folders.find(f => f.id === fid)?.name || fid)
      .join(', ');
    const spaceModeLabel = {
      'none': 'なし',
      'jp': '和文のみ',
      'all': 'すべて'
    }[item.space_mode] || item.space_mode;
    const normalizeLabel = getNormalizeLabel(item.normalize_mode || 'exact');
    const safeQuery = escapeHtml(item.query);
    const safeFolders = escapeHtml(folderNames);
    const safeMode = escapeHtml(item.mode);
    const safeSpaceMode = escapeHtml(spaceModeLabel);
    const safeNormalize = escapeHtml(normalizeLabel);

    return `
      <div class="history-item ${item.pinned ? 'pinned' : ''}" data-id="${item.id}">
        <div class="history-header">
          <div class="history-query">${safeQuery}</div>
          <div class="history-actions">
            <button type="button" class="chip-btn pin-btn" data-id="${item.id}" title="${item.pinned ? 'ピン解除' : 'ピン留め'}">
              <svg viewBox="0 0 24 24" fill="${item.pinned ? 'currentColor' : 'none'}" stroke="currentColor" stroke-width="2">
                <path d="M12 2v8m0 0l-4.5 4.5M12 10l4.5 4.5M12 22v-10"/>
              </svg>
            </button>
            <button type="button" class="chip-btn delete-btn" data-id="${item.id}" title="削除">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
              </svg>
            </button>
          </div>
        </div>
        <div class="history-meta">
          <span class="chip">${safeMode}</span>
          ${item.range_limit > 0 ? `<span class="chip">範囲: ${item.range_limit}</span>` : ''}
          <span class="chip">空白: ${safeSpaceMode}</span>
          <span class="chip">表記ゆれ: ${safeNormalize}</span>
          <span class="chip">${item.result_count} 件</span>
          <span class="chip subtle">${formatTimestamp(item.timestamp)}</span>
        </div>
        <div class="history-folders">${safeFolders}</div>
      </div>
    `;
  }).join('');

  historyListContent.innerHTML = `<div class="history-list">${items}</div>`;

  // Bind event handlers
  historyListContent.querySelectorAll('.history-item').forEach(item => {
    const id = item.dataset.id;
    const historyItem = state.queryHistory.find(h => h.id === id);
    if (!historyItem) return;

    item.addEventListener('click', (e) => {
      if (e.target.closest('.pin-btn') || e.target.closest('.delete-btn')) return;
      executeHistorySearch(historyItem);
    });
  });

  historyListContent.querySelectorAll('.pin-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      togglePinHistory(btn.dataset.id);
      renderHistoryList();
    });
  });

  historyListContent.querySelectorAll('.delete-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      if (confirm('この履歴を削除しますか？')) {
        deleteHistoryItem(btn.dataset.id);
        renderHistoryList();
      }
    });
  });
};

historyToggle.addEventListener('click', openHistoryModal);
historyModal.addEventListener('click', (e) => {
  if (e.target === historyModal) closeHistoryModal();
});
historyModalClose.addEventListener('click', closeHistoryModal);
clearHistoryBtn.addEventListener('click', () => {
  if (confirm('ピン留め以外の履歴をすべて削除しますか？')) {
    clearUnpinnedHistory();
    renderHistoryList();
  }
});

const groupResultsByFile = (results) => {
  const grouped = new Map();
  results.forEach((hit, idx) => {
    const hitIndex = hit._idx ?? idx;
    const existing = grouped.get(hit.path) || {
      path: hit.path,
      displayPath: hit.displayPath || hit.path,
      file: hit.file,
      folderName: hit.folderName || hit.folderId,
      hits: [],
      pages: new Set(),
    };
    if (hit.page && hit.page !== '-') existing.pages.add(hit.page);
    existing.hits.push({ ...hit, _idx: hitIndex });
    grouped.set(hit.path, existing);
  });

  return Array.from(grouped.values()).map(g => ({
    ...g,
    pages: Array.from(g.pages).sort((a, b) => a - b),
  }));
};

// ═══════════════════════════════════════════════════════════════
// HIGHLIGHT TEXT
// ═══════════════════════════════════════════════════════════════

const escapeRegex = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const buildFlexibleRegex = (keyword) => {
  const compact = keyword.replace(/\s+/g, '');
  const chars = Array.from(compact);
  if (chars.length < 2 || chars.length > 64) return null;
  const pattern = chars.map(ch => escapeRegex(ch)).join('\\s*');
  return new RegExp(pattern, 'gi');
};

const collectMatches = (text, regex) => {
  const matches = [];
  if (!regex) return matches;
  regex.lastIndex = 0;
  let match = regex.exec(text);
  while (match) {
    if (!match[0]) break;
    matches.push({ start: match.index, end: match.index + match[0].length });
    if (regex.lastIndex === match.index) regex.lastIndex += 1;
    match = regex.exec(text);
  }
  return matches;
};

const hasOverlap = (ranges, candidate) => (
  ranges.some((range) => !(candidate.end <= range.start || candidate.start >= range.end))
);

const addMatchRanges = (ranges, candidates) => {
  candidates.forEach((candidate) => {
    if (!hasOverlap(ranges, candidate)) {
      ranges.push(candidate);
    }
  });
};

const renderHighlighted = (text, ranges) => {
  if (!ranges.length) return escapeHtml(text);
  const sorted = ranges.slice().sort((a, b) => a.start - b.start);
  let output = '';
  let cursor = 0;
  sorted.forEach((range) => {
    output += escapeHtml(text.slice(cursor, range.start));
    output += `<span class="highlight">${escapeHtml(text.slice(range.start, range.end))}</span>`;
    cursor = range.end;
  });
  output += escapeHtml(text.slice(cursor));
  return output;
};

const highlightText = (text, keywords) => {
  const rawText = String(text ?? '');
  if (!keywords?.length) return escapeHtml(rawText);

  const ranges = [];
  const sortedKeywords = [...keywords].sort((a, b) => b.length - a.length);
  sortedKeywords.forEach((kw) => {
    if (!kw) return;
    const regex = new RegExp(escapeRegex(kw), 'gi');
    let matches = collectMatches(rawText, regex);
    if (!matches.length && state.spaceMode !== 'none') {
      const flex = buildFlexibleRegex(kw);
      if (flex) {
        matches = collectMatches(rawText, flex);
      }
    }
    addMatchRanges(ranges, matches);
  });
  return renderHighlighted(rawText, ranges);
};

const applyDetailContent = (detailEl, hit) => {
  if (!detailEl || !hit) return;
  const text = hit.detail || hit.context || '';
  detailEl.innerHTML = highlightText(text, state.keywords);
};

const buildDetailCacheKey = (detailKey) => {
  if (!detailKey) return '';
  const pageKey = detailKey.page ?? '';
  const hitPos = detailKey.hit_pos ?? 0;
  return `${detailKey.file_id}:${pageKey}:${hitPos}`;
};

const requestDetail = async (detailKey) => {
  const params = new URLSearchParams({
    file_id: detailKey.file_id,
    page: detailKey.page,
    hit_pos: String(detailKey.hit_pos ?? 0),
  });
  const response = await fetch(`/api/detail?${params.toString()}`);
  if (!response.ok) {
    let message = '詳細の取得に失敗しました';
    try {
      const err = await response.json();
      if (err?.detail) message = err.detail;
    } catch (_) {
      message = '詳細の取得に失敗しました';
    }
    const error = new Error(message);
    error.status = response.status;
    throw error;
  }
  const payload = await response.json();
  return payload.detail || '';
};

const fetchDetailForHit = async (hit, detailEl) => {
  if (!hit?.detail_key || !detailEl) return;
  if (hit.detail) {
    applyDetailContent(detailEl, hit);
    return;
  }
  const cacheKey = buildDetailCacheKey(hit.detail_key);
  if (state.detailCache.has(cacheKey)) {
    hit.detail = state.detailCache.get(cacheKey) || '';
    applyDetailContent(detailEl, hit);
    return;
  }

  detailEl.textContent = '詳細を取得中...';
  let request = state.detailRequests.get(cacheKey);
  if (!request) {
    request = requestDetail(hit.detail_key)
      .then((detail) => {
        state.detailCache.set(cacheKey, detail);
        return detail;
      })
      .finally(() => {
        state.detailRequests.delete(cacheKey);
      });
    state.detailRequests.set(cacheKey, request);
  }

  try {
    const detail = await request;
    hit.detail = detail || '';
    applyDetailContent(detailEl, hit);
  } catch (err) {
    console.warn('detail fetch failed', err);
    if (err?.status === 404) {
      detailEl.textContent = '詳細が見つかりません';
    } else {
      detailEl.textContent = err?.message || '詳細の取得に失敗しました';
    }
  }
};

// ═══════════════════════════════════════════════════════════════
// RENDER RESULTS
// ═══════════════════════════════════════════════════════════════

const renderHitCards = (items) => {
  return items.map((r, idx) => {
    const safeFile = escapeHtml(r.file);
    const safePath = escapeHtml(r.displayPath || r.path);
    const safeFolder = escapeHtml(r.folderName || r.folderId);
    const safePage = escapeHtml(r.page);
    return `
      <div class="result-card" data-hit="${r._idx ?? idx}">
        <div class="result-header">
          <div>
            <div class="result-title">${safeFile}</div>
            <div class="result-path">${safePath}</div>
          </div>
          <div class="result-actions">
            <span class="chip">ページ ${safePage}</span>
            <span class="chip">${safeFolder}</span>
            <button type="button" class="chip-btn copy-btn" data-path="${safePath}">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2"/>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
              </svg>
              コピー
            </button>
          </div>
        </div>
        <div class="result-context">${highlightText(r.context, state.keywords)}</div>
        <div class="result-detail">
          <div class="detail-text">${highlightText(r.detail || r.context, state.keywords)}</div>
          <div class="detail-meta">
            <span>フォルダ: ${safeFolder}</span>
            <span>ページ: ${safePage}</span>
            <span class="truncate">パス: ${safePath}</span>
          </div>
        </div>
      </div>
    `;
  }).join('');
};

const renderFileCards = (groups) => {
  return groups.map(g => {
    const safeFile = escapeHtml(g.file);
    const safePath = escapeHtml(g.displayPath || g.path);
    const safeFolder = escapeHtml(g.folderName);
    const pagesLabel = g.pages.length
      ? (g.pages.length > 8 ? `該当ページ数：${g.pages.length}ページ` : `ページ ${g.pages.join(', ')}`)
      : '';
    const safePagesLabel = escapeHtml(pagesLabel);
    return `
      <div class="result-card grouped" data-path="${safePath}">
        <div class="result-header">
          <div>
            <div class="result-title">${safeFile}</div>
            <div class="result-path">${safePath}</div>
            <div class="expand-hint">クリックでヒット一覧を表示</div>
          </div>
          <div class="result-actions">
            <span class="chip">${g.hits.length} ヒット</span>
            ${g.pages.length ? `<span class="chip">${safePagesLabel}</span>` : ''}
            <span class="chip">${safeFolder}</span>
            <button type="button" class="chip-btn copy-btn" data-path="${safePath}">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2"/>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>
              </svg>
              コピー
            </button>
          </div>
        </div>
        <div class="grouped-hits">
          ${g.hits.map(hit => {
            const safeHitFolder = escapeHtml(hit.folderName || hit.folderId);
            const safeHitPage = escapeHtml(hit.page);
            const safeHitPath = escapeHtml(hit.displayPath || hit.path);
            return `
            <div class="hit-item" data-hit="${hit._idx}">
            <div class="hit-summary">
              <div class="hit-meta">
                <span class="chip">ページ ${safeHitPage}</span>
                <span class="chip subtle">${safeHitFolder}</span>
              </div>
              <div class="mini-snippet">${highlightText(hit.context, state.keywords)}</div>
            </div>
            <div class="result-detail">
                <div class="detail-text">${highlightText(hit.detail || hit.context, state.keywords)}</div>
                <div class="detail-meta">
                  <span>ページ: ${safeHitPage}</span>
                  <span class="truncate">パス: ${safeHitPath}</span>
                </div>
              </div>
            </div>
          `;
          }).join('')}
        </div>
      </div>
    `;
  }).join('');
};

const setExpandHint = (card, expanded) => {
  const hint = card.querySelector('.expand-hint');
  if (!hint) return;
  hint.textContent = expanded ? 'クリックで閉じる' : 'クリックでヒット一覧を表示';
};

const bindResultHandlers = (root) => {
  root.querySelectorAll('.copy-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const ok = await copyTextToClipboard(btn.dataset.path);
      if (!ok) return;
      const originalHTML = btn.innerHTML;
      btn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M20 6L9 17l-5-5"/>
        </svg>
        完了
      `;
      btn.classList.add('positive');
      setTimeout(() => {
        btn.innerHTML = originalHTML;
        btn.classList.remove('positive');
      }, 1500);
    });
  });

  root.querySelectorAll('.result-card').forEach(card => {
    setExpandHint(card, card.classList.contains('expanded'));
    card.addEventListener('click', (e) => {
      if (e.target.closest('.copy-btn')) return;
      if (e.target.closest('.hit-item')) return;
      card.classList.toggle('expanded');
      setExpandHint(card, card.classList.contains('expanded'));
      const hitIndex = Number(card.dataset.hit);
      const detailEl = card.querySelector('.detail-text');
      const hit = state.results[hitIndex];
      if (card.classList.contains('expanded')) {
        fetchDetailForHit(hit, detailEl);
      }
    });
  });

  root.querySelectorAll('.hit-item').forEach(item => {
    item.addEventListener('click', (e) => {
      if (e.target.closest('.copy-btn')) return;
      item.classList.toggle('expanded');
      const card = item.closest('.result-card');
      if (card) {
        card.classList.add('expanded');
        setExpandHint(card, true);
      }
      const hitIndex = Number(item.dataset.hit);
      const detailEl = item.querySelector('.detail-text');
      const hit = state.results[hitIndex];
      if (item.classList.contains('expanded')) {
        fetchDetailForHit(hit, detailEl);
      }
    });
  });
};

const resetRenderState = () => {
  state.renderedCount = 0;
  state.groupedResults = null;
  state.isRenderingBatch = false;
};

const getDisplayResults = () => (
  state.filteredResults ? state.filteredResults : state.results
);

const currentRenderList = () => {
  const displayResults = getDisplayResults();
  if (state.viewMode === 'file') {
    if (!state.groupedResults) state.groupedResults = groupResultsByFile(displayResults);
    return state.groupedResults;
  }
  return displayResults;
};

const appendNextBatch = () => {
  if (state.isRenderingBatch) return;
  const list = currentRenderList();
  if (state.renderedCount >= list.length) return;
  state.isRenderingBatch = true;
  const start = state.renderedCount;
  const end = Math.min(start + 100, list.length);
  const slice = list.slice(start, end);
  const listEl = resultsEl.querySelector('.results-list');
  const html = state.viewMode === 'file'
    ? renderFileCards(slice)
    : renderHitCards(slice);
  const temp = document.createElement('div');
  temp.innerHTML = html;
  bindResultHandlers(temp);
  Array.from(temp.children).forEach(node => listEl.appendChild(node));
  state.renderedCount = end;
  state.isRenderingBatch = false;
};

const handleResultsScroll = () => {
  if (!state.results.length || state.isRenderingBatch) return;
  if (resultsEl.scrollTop + resultsEl.clientHeight >= resultsEl.scrollHeight - 200) {
    appendNextBatch();
  }
};

const renderResults = (payload) => {
  if (payload) {
    const { results, keywords } = payload;
    state.results = (results || []).map((item, idx) => ({ ...item, _idx: idx }));
    state.keywords = keywords || [];
    state.filteredResults = null;
    state.filter.folders = new Set();
    state.filter.extensions = new Set();
    state.detailCache = new Map();
    state.detailRequests = new Map();
    buildFilterOptions();
    renderFilterOptions();
  }

  updateResultCount();

  // Show/hide export button
  if (exportBtn) {
    exportBtn.style.display = state.results.length > 0 ? 'inline-flex' : 'none';
  }

  if (!state.results.length) {
    resultsEl.innerHTML = `
      <div class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="11" cy="11" r="8"/>
          <path d="m21 21-4.35-4.35"/>
          <path d="M8 11h6"/>
        </svg>
        <p class="empty-title">一致なし</p>
        <p class="empty-sub">別のキーワードやフォルダでお試しください。</p>
      </div>
    `;
    return;
  }
  resetRenderState();
  resultsEl.innerHTML = '<div class="results-list"></div>';
  resultsEl.scrollTop = 0;
  appendNextBatch();
  while (resultsEl.scrollHeight <= resultsEl.clientHeight + 40) {
    if (state.renderedCount >= currentRenderList().length) break;
    appendNextBatch();
  }
};

const updateResultCount = () => {
  const total = state.results.length;
  const displayCount = getDisplayResults().length;
  resultCountEl.textContent = state.filteredResults
    ? `${displayCount} / ${total} 件`
    : `${total} 件`;
};

const getExtension = (path) => {
  if (!path) return 'unknown';
  const slashIdx = Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\'));
  const dotIdx = path.lastIndexOf('.');
  if (dotIdx === -1 || dotIdx < slashIdx) return 'unknown';
  return `.${path.slice(dotIdx + 1).toLowerCase()}`;
};

const buildFilterOptions = () => {
  const folderCounts = new Map();
  const folderNames = new Map();
  const folderNameCounts = new Map();
  const extCounts = new Map();
  state.results.forEach((r) => {
    const folderId = r.folderId || 'unknown';
    const folderName = r.folderName || 'unknown';
    folderNames.set(folderId, folderName);
    folderCounts.set(folderId, (folderCounts.get(folderId) || 0) + 1);
    folderNameCounts.set(folderName, (folderNameCounts.get(folderName) || 0) + 1);
    const extKey = getExtension(r.path);
    extCounts.set(extKey, (extCounts.get(extKey) || 0) + 1);
  });
  state.filterOptions.folders = Array.from(folderCounts.entries())
    .map(([id, count]) => {
      const name = folderNames.get(id) || 'unknown';
      const dupCount = folderNameCounts.get(name) || 0;
      const label = dupCount > 1 ? `${name} (${id})` : name;
      return { id, name, label, count };
    })
    .sort((a, b) => (b.count - a.count) || a.label.localeCompare(b.label));
  state.filterOptions.extensions = Array.from(extCounts.entries())
    .map(([name, count]) => ({ name, count }))
    .sort((a, b) => (b.count - a.count) || a.name.localeCompare(b.name));
};

const renderFilterOptions = () => {
  if (!filterFoldersEl || !filterExtensionsEl) return;
  const folderHtml = state.filterOptions.folders.length ? state.filterOptions.folders.map((item, idx) => {
    const checked = state.filter.folders.has(item.id) ? 'checked' : '';
    const safeLabel = escapeHtml(item.label);
    return `
      <label class="filter-item">
        <input type="checkbox" data-value="${item.id}" ${checked}>
        <span>${safeLabel}</span>
        <span class="filter-count">${item.count}</span>
      </label>
    `;
  }).join('') : '<div class="empty-sub">対象なし</div>';
  const extHtml = state.filterOptions.extensions.length ? state.filterOptions.extensions.map((item, idx) => {
    const checked = state.filter.extensions.has(item.name) ? 'checked' : '';
    const safeName = escapeHtml(item.name);
    return `
      <label class="filter-item">
        <input type="checkbox" data-value="${item.name}" ${checked}>
        <span>${safeName}</span>
        <span class="filter-count">${item.count}</span>
      </label>
    `;
  }).join('') : '<div class="empty-sub">対象なし</div>';
  filterFoldersEl.innerHTML = folderHtml;
  filterExtensionsEl.innerHTML = extHtml;
};

const applyFilters = () => {
  const selectedFolders = new Set();
  filterFoldersEl?.querySelectorAll('input[type="checkbox"]:checked').forEach((el) => {
    selectedFolders.add(el.dataset.value);
  });
  state.filter.folders = selectedFolders;

  const selectedExts = new Set();
  filterExtensionsEl?.querySelectorAll('input[type="checkbox"]:checked').forEach((el) => {
    selectedExts.add(el.dataset.value);
  });
  state.filter.extensions = selectedExts;

  const hasFolderFilter = state.filter.folders.size > 0;
  const hasExtFilter = state.filter.extensions.size > 0;

  if (!hasFolderFilter && !hasExtFilter) {
    state.filteredResults = null;
  } else {
    state.filteredResults = state.results.filter((r) => {
      if (hasFolderFilter && !state.filter.folders.has(r.folderId || 'unknown')) return false;
      const extKey = getExtension(r.path);
      if (hasExtFilter && !state.filter.extensions.has(extKey)) return false;
      return true;
    });
  }

  updateResultCount();
  if (!getDisplayResults().length) {
    resultsEl.innerHTML = `
      <div class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="11" cy="11" r="8"/>
          <path d="m21 21-4.35-4.35"/>
          <path d="M8 11h6"/>
        </svg>
        <p class="empty-title">一致なし</p>
        <p class="empty-sub">フィルタ条件を見直してください。</p>
      </div>
    `;
    return;
  }
  resetRenderState();
  resultsEl.innerHTML = '<div class="results-list"></div>';
  resultsEl.scrollTop = 0;
  appendNextBatch();
};

const clearFilters = () => {
  state.filter.folders = new Set();
  state.filter.extensions = new Set();
  state.filteredResults = null;
  renderFilterOptions();
  updateResultCount();
  if (!state.results.length) {
    resultsEl.innerHTML = `
      <div class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="11" cy="11" r="8"/>
          <path d="m21 21-4.35-4.35"/>
          <path d="M8 11h6"/>
        </svg>
        <p class="empty-title">一致なし</p>
        <p class="empty-sub">別のキーワードやフォルダでお試しください。</p>
      </div>
    `;
    return;
  }
  resetRenderState();
  resultsEl.innerHTML = '<div class="results-list"></div>';
  resultsEl.scrollTop = 0;
  appendNextBatch();
};

// ═══════════════════════════════════════════════════════════════
// SEARCH
// ═══════════════════════════════════════════════════════════════

const runSearch = async (evt) => {
  evt?.preventDefault();
  if (state.isSearching) return;

  const query = queryInput.value.trim();
  const rangeVal = parseInt(rangeInput.value || '0', 10);
  const spaceMode = spaceModeSelect?.value || 'jp';
  const normalizeMode = normalizeModeSelect?.value || 'normalized';
  const payload = {
    query,
    mode: state.mode,
    range_limit: state.mode === 'AND' ? rangeVal : 0,
    space_mode: spaceMode,
    normalize_mode: normalizeMode,
    folders: Array.from(state.selected),
  };
  state.spaceMode = spaceMode;
  state.normalizeMode = normalizeMode;

  if (!payload.query) {
    alert('キーワードを入力してください');
    queryInput.focus();
    return;
  }
  if (!payload.folders.length) {
    alert('検索対象フォルダを選択してください');
    return;
  }

  state.isSearching = true;
  resultsEl.innerHTML = `
    <div class="loading-state">
      <div class="loading-spinner"></div>
      <p class="loading-text">検索中...</p>
    </div>
  `;

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || '検索に失敗しました');
    }
    const data = await res.json();

    // Save to query history
    state.currentIndexUuid = data.index_uuid || null;
    const effectiveNormalize = data.normalize_mode || normalizeMode;
    if (normalizeModeSelect && data.normalize_mode && data.normalize_mode !== normalizeModeSelect.value) {
      normalizeModeSelect.value = data.normalize_mode;
    }
    addToQueryHistory(
      query,
      state.mode,
      rangeVal,
      spaceMode,
      effectiveNormalize,
      payload.folders,
      data.count || 0,
      state.currentIndexUuid
    );

    if (data.normalize_mode && data.normalize_mode !== normalizeMode) {
      const requestedLabel = getNormalizeLabel(normalizeMode);
      const effectiveLabel = getNormalizeLabel(data.normalize_mode);
      const message = `表記ゆれ: ${requestedLabel} → ${effectiveLabel}`;
      if (message !== lastNormalizeNotice) {
        showNotice(message);
        lastNormalizeNotice = message;
      }
    }

    renderResults(data);
  } catch (err) {
    resultsEl.innerHTML = `
      <div class="empty-state">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <circle cx="12" cy="12" r="10"/>
          <path d="M15 9l-6 6M9 9l6 6"/>
        </svg>
        <p class="empty-title">エラー</p>
        <p class="empty-sub">${escapeHtml(err.message)}</p>
      </div>
    `;
    console.error(err);
  } finally {
    state.isSearching = false;
  }
};

// ═══════════════════════════════════════════════════════════════
// EVENT LISTENERS
// ═══════════════════════════════════════════════════════════════

themeToggle.addEventListener('click', toggleTheme);

modeGroup.addEventListener('click', (e) => {
  const btn = e.target.closest('.toggle-btn');
  if (btn) setMode(btn.dataset.mode);
});

folderToggle.addEventListener('click', toggleFolderList);

viewToggle.addEventListener('click', (e) => {
  const btn = e.target.closest('.toggle-btn');
  if (btn) setViewMode(btn.dataset.view);
});

resultsEl.addEventListener('scroll', handleResultsScroll);

searchForm.addEventListener('submit', runSearch);
refreshBtn.addEventListener('click', loadFolders);

copyPathsBtn.addEventListener('click', async () => {
  const paths = state.folders
    .filter(f => state.selected.has(f.id))
    .map(f => f.displayPath || f.path)
    .join('\n');
  
  if (!paths) {
    alert('コピー対象がありません');
    return;
  }
  
  const ok = await copyTextToClipboard(paths);
  if (!ok) return;
  const originalHTML = copyPathsBtn.innerHTML;
  copyPathsBtn.innerHTML = `
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M20 6L9 17l-5-5"/>
    </svg>
    完了
  `;
  setTimeout(() => {
    copyPathsBtn.innerHTML = originalHTML;
  }, 1500);
});

folderStatusEl.addEventListener('click', (e) => {
  const item = e.target.closest('.status-item');
  if (!item) return;
  const folderId = item.dataset.id;
  const folder = state.folders.find(f => f.id === folderId);
  if (!folder) return;
  if (!folder.ready) {
    alert('このフォルダは準備中です');
    return;
  }
  openFileModal(folderId, folder.name);
});

// Keyboard shortcut: Ctrl+Enter to search
document.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    runSearch();
  }
});

// Export button
if (exportBtn) {
  exportBtn.addEventListener('click', async () => {
    if (!state.results.length) {
      alert('エクスポートする検索結果がありません');
      return;
    }

    const query = queryInput.value.trim();
    const rangeVal = parseInt(rangeInput.value || '0', 10);
    const spaceMode = spaceModeSelect?.value || 'jp';
    const normalizeMode = normalizeModeSelect?.value || 'normalized';
    const payload = {
      query,
      mode: state.mode,
      range_limit: state.mode === 'AND' ? rangeVal : 0,
      space_mode: spaceMode,
      normalize_mode: normalizeMode,
      folders: Array.from(state.selected),
    };

    try {
      exportBtn.disabled = true;
      const originalHTML = exportBtn.innerHTML;
      exportBtn.innerHTML = '<span class="loading-spinner" style="width: 14px; height: 14px; border-width: 2px;"></span> エクスポート中...';

      const res = await fetch('/api/export?format=csv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'エクスポートに失敗しました');
      }

      // Download the file
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `search_results_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);

      exportBtn.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M20 6L9 17l-5-5"/>
        </svg>
        完了
      `;
      setTimeout(() => {
        exportBtn.innerHTML = originalHTML;
      }, 2000);
    } catch (err) {
      alert(`エクスポートエラー: ${err.message}`);
      console.error(err);
      exportBtn.innerHTML = originalHTML;
    } finally {
      exportBtn.disabled = false;
    }
  });
}

if (filterBtn && filterPanel) {
  filterBtn.addEventListener('click', () => {
    const show = filterPanel.style.display === 'none';
    filterPanel.style.display = show ? 'block' : 'none';
    if (show) renderFilterOptions();
  });
}

if (closeFilterBtn && filterPanel) {
  closeFilterBtn.addEventListener('click', () => {
    filterPanel.style.display = 'none';
  });
}

if (applyFilterBtn) {
  applyFilterBtn.addEventListener('click', applyFilters);
}

if (clearFilterBtn) {
  clearFilterBtn.addEventListener('click', clearFilters);
}

// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════

window.addEventListener('DOMContentLoaded', () => {
  initTheme();
  loadQueryHistory();
  initHeartbeat();
  setMode('AND');
  loadFolders();
  queryInput.focus();
  updateFolderToggleLabel();
});
