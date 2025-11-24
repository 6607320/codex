// Глобальное хранилище индекса
let INDEX_DATA = null;

document.addEventListener("DOMContentLoaded", async () => {
  await loadIndex();
  router();
});

async function loadIndex() {
  try {
    const response = await fetch("index.json?" + new Date().getTime());
    INDEX_DATA = await response.json();
  } catch (e) {
    document.body.innerHTML = "<h1>Ошибка: index.json не найден</h1>";
  }
}

function router() {
  const params = new URLSearchParams(window.location.search);
  if (params.has("id")) {
    loadAndRenderQuest(params.get("id"));
  } else {
    renderCodexPage();
  }
}

function renderCodexPage() {
  const app = document.getElementById("app");
  const questsByPart = INDEX_DATA.quests.reduce((acc, quest) => {
    if (!acc[quest.partNumber]) acc[quest.partNumber] = [];
    acc[quest.partNumber].push(quest);
    return acc;
  }, {});

  let html = `<div class="home-scroll-container fade-in">`;
  Object.keys(questsByPart)
    .sort((a, b) => parseInt(a) - parseInt(b))
    .forEach((partKey) => {
      const questsInPart = questsByPart[partKey];
      const scrolls = {};
      questsInPart.forEach((q) => {
        if (!scrolls[q.scrollTitle]) scrolls[q.scrollTitle] = [];
        scrolls[q.scrollTitle].push(q);
      });
      html += `<div class="part-container"><h2>${questsInPart[0].partTitle}</h2>`;
      for (const scrollTitle in scrolls) {
        html += `<div class="scroll-container"><h3>${scrollTitle}</h3><div class="quest-grid">`;
        scrolls[scrollTitle]
          .sort((a, b) => parseFloat(a.id) - parseFloat(b.id))
          .forEach((q) => {
            const clickAttr =
              q.status === "active"
                ? `onclick="window.location.href='quest.html?id=${q.id}'"`
                : "";
            html += `<div class="quest-card ${q.status}" ${clickAttr}><div class="q-id">#${q.id}</div><div class="q-title">${q.title}</div></div>`;
          });
        html += `</div></div>`;
      }
      html += `</div>`;
    });
  html += `</div>`;
  app.innerHTML = html;
}

async function loadAndRenderQuest(id) {
  const app = document.getElementById("app");
  app.innerHTML = "<h2>Загрузка...</h2>";
  try {
    const response = await fetch(`quests/quest_${id}.json?` + new Date().getTime());
    if (!response.ok) throw new Error("File not found");
    const questData = await response.json();
    renderQuestView(questData);
  } catch (e) {
    app.innerHTML = "<h2>Ошибка загрузки квеста</h2>";
  }
}

// ... (остальные функции)

function renderQuestView(quest) {
  // quest теперь это questData
  const app = document.getElementById("app");
  app.innerHTML = `
        <div class="quest-layout fade-in">
            <div class="col col-legend"><div class="col-header">📜 ЛЕГЕНДА</div><div class="scroll-content markdown-body"><h2>${quest.id} ${quest.title}</h2>${quest.legend}</div></div>
            <div class="col col-terminal"><div class="col-header">💻 ТЕРМИНАЛ</div><div class="terminal-window" id="term-window"><div id="term-output"><div class="sys-msg">Codex OS v1.0 loaded...</div><div class="sys-msg">Target: Quest ${quest.id}</div><br></div><div class="input-line"><span class="prompt">mage@codex:~$</span><input type="text" id="term-input"></div></div></div>
            <div class="col col-manifest"><div class="col-header">📦 МАНИФЕСТ</div><div class="scroll-content markdown-body">${quest.manifest}</div></div>
        </div>`;

  // === ВСТАВИТЬ ЭТОТ БЛОК СЮДА ===
  // Находим все ссылки внутри Легенды и Манифеста
  const links = app.querySelectorAll(".markdown-body a");
  links.forEach((link) => {
    // Если ссылка внешняя (начинается с http), открываем в новой вкладке
    if (link.href.startsWith("http")) {
      link.target = "_blank";
      link.rel = "noopener noreferrer"; // Правило безопасности
    }
  });
  // ===============================

  // Передаем весь объект questData в initTerminalLogic
  initTerminalLogic(quest);
}

// ... (остальные функции остаются без изменений) ...

function initTerminalLogic(quest) {
  const input = document.getElementById("term-input");
  const output = document.getElementById("term-output");
  const win = document.getElementById("term-window");

  if (!input) return;

  input.focus();
  input.addEventListener("keypress", async (e) => {
    // Добавляем async
    if (e.key === "Enter") {
      const cmd = input.value.trim();
      if (cmd === "") return;

      // Создаем новую строку-контейнер
      const line = document.createElement("div");

      // Создаем спан для промпта
      const promptSpan = document.createElement("span");
      promptSpan.className = "prompt";
      promptSpan.textContent = "mage@codex:~$";

      // Собираем строку: сначала промпт, потом текст команды
      line.appendChild(promptSpan);
      line.appendChild(document.createTextNode(` ${cmd}`));

      // Безопасно добавляем строку в вывод терминала
      output.appendChild(line);
      input.value = "";
      input.disabled = true; // Блокируем ввод, пока идет "печать"

      const scenarioStep = (quest.scenario || []).find((step) =>
        cmd.startsWith(step.command),
      );

      if (scenarioStep) {
        // Запускаем эффект "печатной машинки" и ждем его завершения
        await typewriterEffect(output, scenarioStep.output, win);

        if (scenarioStep.is_final) {
          showNextButton(quest, output);
        }
      } else {
        await typewriterEffect(output, "Команда не распознана.", win);
      }

      output.innerHTML += `<br>`;
      win.scrollTop = win.scrollHeight;
      input.disabled = false; // Разблокируем ввод
      input.focus();
    }
  });
}

// --- НОВАЯ ФУНКЦИЯ ДЛЯ ЭФФЕКТА "ПЕЧАТНОЙ МАШИНКИ" ---
async function typewriterEffect(container, text, scrollContainer) {
  const lines = text.split("\n");
  for (const line of lines) {
    let lineDiv = document.createElement("div");
    lineDiv.className = "sys-msg"; // Используем наш стиль для вывода
    container.appendChild(lineDiv);

    for (let i = 0; i < line.length; i++) {
      lineDiv.innerHTML += line.charAt(i);
      scrollContainer.scrollTop = scrollContainer.scrollHeight; // Прокручиваем вниз
      // Задержка. 15ms - быстро, 30ms - средне.
      // Для очень длинных логов можно сделать меньше (например, 5ms).
      await new Promise((resolve) => setTimeout(resolve, 15));
    }
  }
}

function showNextButton(quest, outputContainer) {
  // ... (эта функция остается без изменений)
  const sortedQuests = [...INDEX_DATA.quests].sort(
    (a, b) => parseFloat(a.id) - parseFloat(b.id),
  );
  const currentIndex = sortedQuests.findIndex((q) => q.id === quest.id);

  if (currentIndex !== -1 && currentIndex < sortedQuests.length - 1) {
    const nextQuest = sortedQuests[currentIndex + 1];
    outputContainer.innerHTML += `<div style="text-align: center; margin: 25px 0 15px 0;"><a href="quest.html?id=${nextQuest.id}" class="btn-magic">К следующему квесту: #${nextQuest.id} →</a></div>`;
  } else {
    outputContainer.innerHTML += `<div class="success-msg" style="text-align:center; margin-top: 20px;">✨ Вы освоили последний свиток! ✨</div>`;
  }
}
