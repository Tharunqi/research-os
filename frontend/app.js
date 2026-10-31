(() => {
  let mathTypesetQueue = Promise.resolve();

  const ACTIONS = {
    ask: { label: "Ask about a paper", placeholder: "Ask a focused question about a paper…", working: "Retrieving evidence from the paper" },
    search: { label: "Find papers", placeholder: "Enter a paper title, arXiv ID, or research topic…", working: "Searching and indexing papers" },
    summarize: { label: "Summarize a paper", placeholder: "Enter the paper title or arXiv ID…", working: "Reading and summarizing the paper" },
    extract: { label: "Extract results", placeholder: "Enter a paper title and optional focus…", working: "Extracting the reported results" },
    compare: { label: "Compare papers", placeholder: "Enter two or more paper titles…", working: "Preparing the papers for comparison" },
    synthesize: { label: "Synthesize literature", placeholder: "Enter a topic and the papers to synthesize…", working: "Synthesizing evidence across papers" },
  };

  const state = {
    projects: [],
    project: null,
    chats: [],
    chat: null,
    messages: [],
    papers: [],
    busy: false,
    projectRequestId: 0,
    toastTimer: null,
  };

  const $ = (selector) => document.querySelector(selector);
  const els = {
    projects: $("#project-list"),
    chats: $("#chat-list"),
    papers: $("#paper-list"),
    messages: $("#message-list"),
    welcome: $("#welcome"),
    projectDialog: $("#project-dialog"),
    projectForm: $("#project-form"),
    projectName: $("#project-name"),
    projectDescription: $("#project-description"),
    composer: $("#composer"),
    input: $("#message-input"),
    mode: $("#action-mode"),
    send: $("#send-button"),
    newChat: $("#new-chat-button"),
    topNewChat: $("#top-new-chat"),
    breadcrumb: $("#breadcrumb"),
    paperCount: $("#paper-count"),
    libraryDescription: $("#library-description"),
    saveStatus: $("#save-status"),
    refreshPapers: $("#refresh-papers"),
    toast: $("#toast"),
    mobilePicker: $("#mobile-project-picker"),
    mobileMenu: $("#mobile-project-menu"),
    mobileProjects: $("#mobile-project-list"),
    mobileChatPicker: $("#mobile-chat-picker"),
    mobileChatMenu: $("#mobile-chat-menu"),
    mobileChats: $("#mobile-chat-list"),
    mobileLibraryToggle: $("#mobile-library-toggle"),
    compactLibraryButton: $("#compact-library-button"),
    compactPaperCount: $("#compact-paper-count"),
    libraryPanel: $(".library-panel"),
  };

  async function api(path, options = {}) {
    const response = await fetch(path, {
      ...options,
      headers: {
        ...(options.body ? { "Content-Type": "application/json" } : {}),
        ...options.headers,
      },
    });
    const raw = await response.text();
    let data = null;
    try { data = raw ? JSON.parse(raw) : null; } catch { data = raw; }
    if (!response.ok) {
      const detail = data && typeof data === "object" ? data.detail : null;
      throw new Error(typeof detail === "string" ? detail : `Request failed (${response.status})`);
    }
    return data;
  }

  function showToast(message) {
    window.clearTimeout(state.toastTimer);
    els.toast.textContent = message;
    els.toast.classList.add("visible");
    state.toastTimer = window.setTimeout(() => els.toast.classList.remove("visible"), 3200);
  }

  function setStatus(message = "") {
    els.saveStatus.textContent = message;
  }

  function safeStoredId(key) {
    try { return window.localStorage.getItem(key); } catch { return null; }
  }

  function storeId(key, value) {
    try {
      if (value == null) window.localStorage.removeItem(key);
      else window.localStorage.setItem(key, String(value));
    } catch { /* The app works without persistent browser storage. */ }
  }

  function renderProjects() {
    els.projects.replaceChildren();
    if (!state.projects.length) {
      const empty = document.createElement("div");
      empty.className = "nav-empty";
      empty.textContent = "Create a workspace to start collecting papers.";
      els.projects.append(empty);
    }

    for (const project of state.projects) {
      const button = document.createElement("button");
      button.className = `nav-item${state.project?.id === project.id ? " active" : ""}`;
      button.type = "button";
      button.title = project.name;
      button.innerHTML = '<span class="nav-item-icon">◫</span>';
      const label = document.createElement("span");
      label.className = "nav-item-label";
      label.textContent = project.name;
      button.append(label);
      button.addEventListener("click", () => selectProject(project.id));
      els.projects.append(button);
    }
    renderMobileProjects();
  }

  function renderMobileProjects() {
    els.mobileProjects.replaceChildren();
    for (const project of state.projects) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `mobile-project-option${state.project?.id === project.id ? " active" : ""}`;
      button.textContent = project.name;
      button.addEventListener("click", () => {
        closeMobileMenu();
        selectProject(project.id);
      });
      els.mobileProjects.append(button);
    }
    setMobilePickerLabel(els.mobilePicker, state.project?.name || "Workspaces");
    setMobilePickerLabel(els.mobileChatPicker, state.chat?.title || "Chats");
    els.mobileChats.replaceChildren();
    for (const chat of state.chats) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `mobile-project-option${state.chat?.id === chat.id ? " active" : ""}`;
      button.textContent = chat.title || `Research chat ${chat.id}`;
      button.addEventListener("click", () => {
        closeMobileMenu();
        closeMobileChatMenu();
        selectChat(chat.id);
      });
      els.mobileChats.append(button);
    }
    if (!state.chats.length) {
      const empty = document.createElement("div");
      empty.className = "nav-empty";
      empty.textContent = "No chats in this workspace yet.";
      els.mobileChats.append(empty);
    }
  }

  function setMobilePickerLabel(button, value) {
    const label = document.createElement("span");
    label.className = "mobile-picker-label";
    label.textContent = value;
    const caret = document.createElement("span");
    caret.textContent = "⌄";
    button.replaceChildren(label, caret);
  }

  function renderChats() {
    els.chats.replaceChildren();
    renderMobileProjects();
    if (!state.project) {
      const empty = document.createElement("div");
      empty.className = "nav-empty";
      empty.textContent = "Chats are grouped by workspace.";
      els.chats.append(empty);
      return;
    }
    if (!state.chats.length) {
      const empty = document.createElement("div");
      empty.className = "nav-empty";
      empty.textContent = "No conversations yet.";
      els.chats.append(empty);
      return;
    }
    for (const chat of state.chats) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = `nav-item${state.chat?.id === chat.id ? " active" : ""}`;
      button.title = chat.title || `Research chat ${chat.id}`;
      const icon = document.createElement("span");
      icon.className = "nav-item-icon";
      icon.textContent = "◌";
      const label = document.createElement("span");
      label.className = "nav-item-label";
      label.textContent = chat.title || `Research chat ${chat.id}`;
      button.append(icon, label);
      button.addEventListener("click", () => selectChat(chat.id));
      els.chats.append(button);
    }
  }

  function renderPapers() {
    els.paperCount.textContent = String(state.papers.length);
    els.compactPaperCount.textContent = String(state.papers.length);
    els.mobileLibraryToggle.textContent = `▤ View paper library (${state.papers.length})`;
    els.papers.replaceChildren();
    if (!state.project) {
      els.libraryDescription.textContent = "Select a workspace to view its papers.";
    } else {
      els.libraryDescription.textContent = state.project.description || "Papers discovered in this workspace will appear here.";
    }
    if (!state.papers.length) {
      const empty = document.createElement("div");
      empty.className = "empty-library";
      empty.textContent = state.project ? "No saved papers yet. Use Find papers to discover and index research." : "Your saved papers will live here.";
      els.papers.append(empty);
      return;
    }
    for (const paper of state.papers) {
      const card = document.createElement("article");
      card.className = "paper-card";
      const title = document.createElement("a");
      title.className = "paper-title";
      title.textContent = paper.title || "Untitled paper";
      const titleUrl = safeWebUrl(paper.paper_url || paper.pdf_url);
      if (titleUrl) {
        title.href = titleUrl;
        title.target = "_blank";
        title.rel = "noreferrer";
      } else {
        title.removeAttribute("href");
      }
      const meta = document.createElement("div");
      meta.className = "paper-meta";
      const authorText = Array.isArray(paper.authors) ? paper.authors.slice(0, 2).join(", ") : (paper.authors || "");
      const published = paper.published ? String(paper.published).slice(0, 4) : "";
      meta.textContent = [paper.arxiv_id ? `arXiv:${paper.arxiv_id}` : "", published, authorText].filter(Boolean).join(" · ") || "Saved to workspace";
      card.append(title, meta);
      if (paper.abstract) {
        const abstract = document.createElement("p");
        abstract.className = "paper-abstract";
        abstract.textContent = paper.abstract;
        card.append(abstract);
      }
      const links = document.createElement("div");
      links.className = "paper-links";
      addPaperLink(links, "Paper", paper.paper_url);
      addPaperLink(links, "PDF", paper.pdf_url);
      if (links.childElementCount) card.append(links);
      els.papers.append(card);
    }
  }

  function addPaperLink(container, label, rawUrl) {
    const url = safeWebUrl(rawUrl);
    if (!url) return;
    const link = document.createElement("a");
    link.className = "paper-link";
    link.href = url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.textContent = `${label} ↗`;
    container.append(link);
  }

  function safeWebUrl(rawUrl) {
    if (!rawUrl) return null;
    try {
      const url = new URL(rawUrl, window.location.origin);
      return ["http:", "https:"].includes(url.protocol) ? url.href : null;
    } catch { return null; }
  }

  function renderMessages() {
    const hasMessages = state.messages.length > 0;
    els.welcome.hidden = hasMessages;
    els.messages.hidden = !hasMessages;
    if (window.MathJax?.typesetClear) {
      window.MathJax.typesetClear([els.messages]);
    }
    els.messages.replaceChildren();
    for (const message of state.messages) {
      const row = document.createElement("article");
      row.className = `message-row ${message.role === "user" ? "user" : "assistant"}`;
      if (message.role !== "user") {
        const avatar = document.createElement("span");
        avatar.className = "message-avatar";
        avatar.textContent = "R";
        row.append(avatar);
      }
      const content = document.createElement("div");
      content.className = "message-content";
      if (message._working) {
        content.innerHTML = `<span class="typing-indicator">${escapeHtml(message.workingText || "Working on your request")} <span class="typing-dots"><i></i><i></i><i></i></span></span>`;
      } else {
        content.innerHTML = renderMarkdown(message.content || "");
      }
      row.append(content);
      if (message.role !== "user" && Array.isArray(message.sources) && message.sources.length) {
        const sources = document.createElement("div");
        sources.className = "message-sources";
        for (const paper of message.sources) {
          const source = document.createElement("a");
          source.className = "source-chip";
          source.textContent = paper.title || paper.arxiv_id || "Paper";
          source.title = paper.title || "Open paper";
          const sourceUrl = safeWebUrl(paper.paper_url || paper.pdf_url);
          source.href = sourceUrl || "#";
          if (sourceUrl) {
            source.target = "_blank";
            source.rel = "noreferrer";
          }
          sources.append(source);
        }
        content.append(sources);
      }
      els.messages.append(row);
    }
    typesetMessages();
    els.messages.scrollTop = els.messages.scrollHeight;
    els.messages.lastElementChild?.scrollIntoView({ block: "end", behavior: "smooth" });
  }

  function typesetMessages() {
    if (!window.MathJax?.typesetPromise) return;
    mathTypesetQueue = mathTypesetQueue
      .catch(() => {})
      .then(() => window.MathJax.typesetPromise([els.messages]))
      .catch(() => {});
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
  }

  function inlineMarkdown(value) {
    let output = value.replace(/`([^`]+)`/g, "<code>$1</code>");
    output = output.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    output = output.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    output = output.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, (_match, label, href) => {
      try {
        const parsed = new URL(href);
        if (!["http:", "https:"].includes(parsed.protocol)) return label;
        return `<a href="${escapeHtml(parsed.href)}" target="_blank" rel="noreferrer">${label}</a>`;
      } catch { return label; }
    });
    return output;
  }

  function extractMath(value) {
    let text = escapeHtml(value);
    const entries = [];
    const patterns = [
      { pattern: /\$\$[\s\S]+?\$\$/g, display: true },
      { pattern: /\\\[[\s\S]+?\\\]/g, display: true },
      { pattern: /\\begin\{(equation\*?|align\*?|aligned|gather\*?|gathered|multline\*?|cases)\}[\s\S]*?\\end\{\1\}/g, display: true },
      { pattern: /\\\([\s\S]+?\\\)/g, display: false },
      { pattern: /\$[^$\n]+?\$/g, display: false },
    ];

    for (const { pattern, display } of patterns) {
      text = text.replace(pattern, (source) => {
        const marker = `\uE000${entries.length}\uE001`;
        const environment = source.match(/^\\begin\{([^}]+)\}/)?.[1];
        const needsMathWrapper = ["aligned", "gathered", "cases"].includes(environment);
        entries.push({
          marker,
          source: needsMathWrapper ? `$$${source}$$` : source,
          display,
        });
        return marker;
      });
    }
    return { text: text.replace(/\r/g, ""), entries };
  }

  function restoreMath(value, entries) {
    let output = value;
    for (const entry of entries) {
      output = output.split(entry.marker).join(entry.source);
    }
    return output;
  }

  function renderMarkdown(value) {
    const math = extractMath(value);
    const mathByMarker = new Map(math.entries.map((entry) => [entry.marker, entry]));
    const lines = math.text.split("\n");
    const output = [];
    let list = [];
    const flushList = () => {
      if (!list.length) return;
      output.push(`<ul>${list.map((line) => `<li>${restoreMath(inlineMarkdown(line), math.entries)}</li>`).join("")}</ul>`);
      list = [];
    };
    for (const line of lines) {
      const standaloneMath = mathByMarker.get(line.trim());
      if (standaloneMath?.display) {
        flushList();
        output.push(`<div class="math-block">${standaloneMath.source}</div>`);
        continue;
      }
      const item = line.match(/^\s*[-*]\s+(.+)$/);
      if (item) { list.push(item[1]); continue; }
      flushList();
      if (!line.trim()) continue;
      const heading = line.match(/^#{1,3}\s+(.+)$/);
      if (heading) output.push(`<p><strong>${restoreMath(inlineMarkdown(heading[1]), math.entries)}</strong></p>`);
      else output.push(`<p>${restoreMath(inlineMarkdown(line), math.entries)}</p>`);
    }
    flushList();
    return output.join("") || "<p></p>";
  }

  function setComposerEnabled(enabled) {
    const ready = enabled && !state.busy;
    els.input.disabled = !ready;
    els.mode.disabled = !ready;
    els.send.disabled = !ready || !els.input.value.trim();
    els.newChat.disabled = !state.project || state.busy;
    els.topNewChat.disabled = !state.project || state.busy;
    els.refreshPapers.disabled = !state.project;
    els.input.placeholder = state.project ? ACTIONS[els.mode.value].placeholder : "Select a workspace to begin…";
  }

  function setBusy(busy) {
    state.busy = busy;
    setComposerEnabled(Boolean(state.project && state.chat));
    setStatus(busy ? "Researching…" : "");
  }

  async function loadProjects() {
    const data = await api("/projects");
    state.projects = Array.isArray(data?.projects) ? data.projects : [];
    renderProjects();
    const saved = safeStoredId("researchos.projectId");
    const initial = state.projects.find((project) => String(project.id) === saved) || state.projects[0];
    if (initial) await selectProject(initial.id);
    else {
      state.project = null;
      state.chats = [];
      state.chat = null;
      state.papers = [];
      renderProjects();
      renderChats();
      renderPapers();
      setComposerEnabled(false);
    }
  }

  async function selectProject(projectId) {
    if (state.busy) return showToast("Wait for the current research request to finish first.");
    const project = state.projects.find((item) => item.id === Number(projectId));
    if (!project) return;
    const requestId = ++state.projectRequestId;
    state.project = project;
    state.chat = null;
    state.chats = [];
    state.messages = [];
    state.papers = [];
    storeId("researchos.projectId", project.id);
    els.breadcrumb.innerHTML = `<strong>${escapeHtml(project.name)}</strong>`;
    renderProjects();
    renderChats();
    renderPapers();
    renderMessages();
    setComposerEnabled(false);
    try {
      const savedChatId = safeStoredId("researchos.chatId");
      const [, chats] = await Promise.all([loadPapers(), api(`/projects/${project.id}/chats`)]);
      if (requestId !== state.projectRequestId || state.project?.id !== project.id) return;
      state.chats = Array.isArray(chats) ? chats : [];
      renderChats();
      const selected = state.chats.find((chat) => String(chat.id) === savedChatId) || state.chats[0];
      if (selected) await selectChat(selected.id);
      else await createChat();
    } catch (error) {
      showToast(error.message || "Could not load this workspace.");
      setComposerEnabled(false);
    }
  }

  async function loadPapers() {
    if (!state.project) return;
    const projectId = state.project.id;
    const result = await api(`/projects/${projectId}/papers`);
    if (state.project?.id !== projectId) return;
    state.papers = Array.isArray(result?.papers) ? result.papers : [];
    renderPapers();
  }

  async function selectChat(chatId) {
    if (state.busy) return showToast("Wait for the current research request to finish first.");
    if (!state.project) return;
    const chat = state.chats.find((item) => item.id === Number(chatId));
    if (!chat) return;
    state.chat = chat;
    state.messages = [];
    storeId("researchos.chatId", chat.id);
    renderChats();
    els.breadcrumb.innerHTML = `<strong>${escapeHtml(state.project.name)}</strong><span> / </span>${escapeHtml(chat.title || `Research chat ${chat.id}`)}`;
    renderMessages();
    setComposerEnabled(true);
    try {
      const messages = await api(`/chats/${chat.id}/messages`);
      if (state.chat?.id !== chat.id) return;
      state.messages = Array.isArray(messages) ? messages : [];
      renderMessages();
    } catch (error) {
      showToast(error.message || "Could not load this chat.");
    }
  }

  async function createChat() {
    if (!state.project || state.busy) return;
    try {
      const projectId = state.project.id;
      const chat = await api(`/projects/${projectId}/chats`, {
        method: "POST",
        body: JSON.stringify({ title: null }),
      });
      if (state.project?.id !== projectId) return;
      state.chats.unshift(chat);
      renderChats();
      await selectChat(chat.id);
      els.input.focus();
    } catch (error) {
      showToast(error.message || "Could not create a chat.");
    }
  }

  async function createProject(event) {
    event.preventDefault();
    const name = els.projectName.value.trim();
    if (!name) return;
    const submit = $("#create-project-submit");
    submit.disabled = true;
    try {
      const project = await api("/projects", {
        method: "POST",
        body: JSON.stringify({ name, description: els.projectDescription.value.trim() || null }),
      });
      state.projects.unshift(project);
      els.projectDialog.close();
      els.projectForm.reset();
      await selectProject(project.id);
      showToast("Workspace created.");
    } catch (error) {
      showToast(error.message || "Could not create the workspace.");
    } finally {
      submit.disabled = false;
    }
  }

  async function submitResearch(event) {
    event.preventDefault();
    const draft = els.input.value.trim();
    if (!draft || !state.project || !state.chat || state.busy) return;
    const action = els.mode.value;
    const question = phraseForMode(action, draft);
    const placeholder = els.input.placeholder;
    state.messages.push({ role: "user", content: question });
    state.messages.push({ role: "assistant", content: "", _working: true, workingText: ACTIONS[action].working });
    els.input.value = "";
    resizeInput();
    renderMessages();
    setBusy(true);
    try {
      const result = await api("/research", {
        method: "POST",
        body: JSON.stringify({ question, project_id: state.project.id, session_id: state.chat.id }),
      });
      const workingIndex = state.messages.findIndex((message) => message._working);
      if (workingIndex >= 0) {
        state.messages[workingIndex] = {
          role: "assistant",
          content: result?.answer || "Research finished without an answer.",
          sources: Array.isArray(result?.papers) ? result.papers : [],
        };
      }
      renderMessages();
      try { await loadPapers(); } catch { showToast("Answer saved, but the paper list could not refresh."); }
      setStatus("Saved to chat");
      window.setTimeout(() => setStatus(""), 2000);
    } catch (error) {
      state.messages = state.messages.filter((message) => !message._working);
      renderMessages();
      showToast(error.message || "Research request failed.");
      try {
        const savedMessages = await api(`/chats/${state.chat.id}/messages`);
        state.messages = Array.isArray(savedMessages) ? savedMessages : state.messages;
        renderMessages();
      } catch { /* Keep the local conversation visible if history cannot reload. */ }
    } finally {
      setBusy(false);
      els.input.placeholder = placeholder;
      els.input.focus();
    }
  }

  function resizeInput() {
    els.input.style.height = "auto";
    els.input.style.height = `${Math.min(els.input.scrollHeight, 150)}px`;
    els.send.disabled = !els.input.value.trim() || !state.project || !state.chat || state.busy;
  }

  function setMode(mode) {
    if (!ACTIONS[mode]) return;
    els.mode.value = mode;
    els.input.placeholder = state.project ? ACTIONS[mode].placeholder : "Select a workspace to begin…";
  }

  function phraseForMode(mode, draft) {
    const patterns = {
      ask: /$^/,
      search: /\b(find|search|discover|recommend|add|save|look for)\b.{0,50}\b(papers?|articles?)\b|\b(papers?|articles?)\b.{0,35}\b(on|about|for)\b/i,
      summarize: /\b(summarize|summarise|summary|overview|tl\s*;?\s*dr)\b/i,
      extract: /\b(extract|results?|metrics|datasets|benchmarks|baselines|ablation|tables?|performance|quantitative|scores?)\b/i,
      compare: /\b(compare|comparison|contrast|versus|vs\.?|differences?|side[-\s]?by[-\s]?side)\b|\bdifference between\b/i,
      synthesize: /\b(literature review|literature survey|related work|synthesi[sz]e?|synthesis|state of the art)\b/i,
    };
    if (!mode || mode === "ask" || patterns[mode]?.test(draft)) return draft;
    const prefixes = {
      search: "Find papers about: ",
      summarize: "Summarize this paper: ",
      extract: "Extract the results from this paper: ",
      compare: "Compare these papers: ",
      synthesize: "Synthesize the literature across these papers: ",
    };
    return `${prefixes[mode] || ""}${draft}`;
  }

  function openProjectDialog() {
    closeMobileMenu();
    els.projectDialog.showModal();
    window.setTimeout(() => els.projectName.focus(), 40);
  }

  function closeMobileMenu() {
    els.mobileMenu.hidden = true;
    els.mobilePicker.setAttribute("aria-expanded", "false");
  }

  function closeMobileChatMenu() {
    els.mobileChatMenu.hidden = true;
    els.mobileChatPicker.setAttribute("aria-expanded", "false");
  }

  function closeMobileLibrary() {
    els.libraryPanel.classList.remove("mobile-open");
    els.libraryPanel.classList.remove("tablet-open");
  }

  function bindEvents() {
    $("#new-project-button").addEventListener("click", openProjectDialog);
    $("#mobile-new-project").addEventListener("click", openProjectDialog);
    $("#cancel-project").addEventListener("click", () => els.projectDialog.close());
    els.projectForm.addEventListener("submit", createProject);
    els.newChat.addEventListener("click", createChat);
    els.topNewChat.addEventListener("click", createChat);
    els.refreshPapers.addEventListener("click", () => loadPapers().catch((error) => showToast(error.message)));
    els.composer.addEventListener("submit", submitResearch);
    els.input.addEventListener("input", resizeInput);
    els.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        els.composer.requestSubmit();
      }
    });
    els.mode.addEventListener("change", () => setMode(els.mode.value));
    $("#starter-grid").addEventListener("click", (event) => {
      const card = event.target.closest("[data-mode]");
      if (!card) return;
      setMode(card.dataset.mode);
      if (!state.project) openProjectDialog();
      else els.input.focus();
    });
    els.mobilePicker.addEventListener("click", () => {
      const open = els.mobileMenu.hidden;
      closeMobileChatMenu();
      closeMobileLibrary();
      els.mobileMenu.hidden = !open;
      els.mobilePicker.setAttribute("aria-expanded", String(open));
    });
    els.mobileChatPicker.addEventListener("click", () => {
      const open = els.mobileChatMenu.hidden;
      closeMobileMenu();
      closeMobileLibrary();
      els.mobileChatMenu.hidden = !open;
      els.mobileChatPicker.setAttribute("aria-expanded", String(open));
    });
    els.mobileLibraryToggle.addEventListener("click", () => {
      closeMobileMenu();
      els.libraryPanel.classList.add("mobile-open");
    });
    els.compactLibraryButton.addEventListener("click", () => {
      const open = !els.libraryPanel.classList.contains("tablet-open");
      closeMobileLibrary();
      if (open) els.libraryPanel.classList.add("tablet-open");
    });
    $("#close-mobile-library").addEventListener("click", closeMobileLibrary);
    document.addEventListener("click", (event) => {
      if (!els.mobileMenu.hidden && !els.mobileMenu.contains(event.target) && !els.mobilePicker.contains(event.target)) closeMobileMenu();
      if (!els.mobileChatMenu.hidden && !els.mobileChatMenu.contains(event.target) && !els.mobileChatPicker.contains(event.target)) closeMobileChatMenu();
      const libraryOpen = els.libraryPanel.classList.contains("mobile-open") || els.libraryPanel.classList.contains("tablet-open");
      if (libraryOpen && !els.libraryPanel.contains(event.target) && !els.mobilePicker.contains(event.target) && !els.mobileLibraryToggle.contains(event.target) && !els.compactLibraryButton.contains(event.target)) closeMobileLibrary();
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") { closeMobileMenu(); closeMobileChatMenu(); closeMobileLibrary(); }
    });
  }

  bindEvents();
  setComposerEnabled(false);
  renderMobileProjects();
  if (window.location.protocol === "file:") {
    els.libraryDescription.textContent = "Open the running app at http://localhost:8000/ui/ to connect to your database.";
    $("#welcome-footnote").textContent = "This file is a preview. Start the backend and open http://localhost:8000/ui/ to use your projects and research tools.";
    showToast("For projects and research, open http://localhost:8000/ui/ after starting the backend.");
    return;
  }
  loadProjects().catch((error) => {
    showToast(error.message || "Could not connect to ResearchOS.");
    els.libraryDescription.textContent = "Could not connect to the ResearchOS API.";
  });
})();
