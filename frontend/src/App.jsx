import { useEffect, useMemo, useState, useRef } from 'react';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '';
const API_PREFIX = API_BASE_URL ? API_BASE_URL.replace(/\/$/, '') : '';

const apiUrl = (path) => {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  if (API_PREFIX) {
    return `${API_PREFIX}${normalizedPath}`;
  }
  // Use Vite proxy '/api' locally, but request root endpoints directly in production
  return import.meta.env.PROD ? normalizedPath : `/api${normalizedPath}`;
};

const modeCards = [
  { id: 'ask', name: 'Ask your notes', description: 'Direct cited answers from your notes', tone: 'blue' },
  { id: 'revision', name: 'Revision sheet', description: 'Definition, key points, formulas, traps', tone: 'blue' },
  { id: 'quiz', name: 'Auto quiz', description: 'Exam-style MCQs with explanations', tone: 'amber' },
  { id: 'explain', name: 'Explain simply', description: 'Zero-jargon analogies and memory hooks', tone: 'blue' },
  { id: 'audio', name: 'Audio notes', description: 'Generate portable revision MP3', tone: 'green' },
  { id: 'night', name: 'Night before exam', description: 'Urgent compression for final hours', tone: 'amber pulse' },
  { id: 'flashcards', name: 'Flashcards', description: 'Anki-style front/back drill mode', tone: 'amber' },
  { id: 'highlights', name: 'Smart highlights', description: 'Top scoring sentences for revision', tone: 'blue' },
  { id: 'upload', name: 'Upload source', description: 'Add and index new PDF material', tone: 'green' },
];

const initialModeState = {
  collectionId: '',
  question: '',
  language: 'en',
  topic: '',
  concept: '',
  subject: '',
  examHoursAway: 8,
  numQuestions: 5,
  numCards: 10,
};

function IconWrapper({ children }) {
  return (
    <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {children}
    </svg>
  );
}

function ChatIcon() {
  return <IconWrapper><path d="M14 9a2 2 0 0 1-2 2H6l-4 4V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2z" /><path d="M18 9h2a2 2 0 0 1 2 2v11l-4-4h-6a2 2 0 0 1-2-2v-1" /></IconWrapper>;
}
function BookOpenIcon() {
  return <IconWrapper><path d="M3 6.5c2.5-1 5.5-1 8 0v12c-2.5-1-5.5-1-8 0z" /><path d="M13 6.5c2.5-1 5.5-1 8 0v12c-2.5-1-5.5-1-8 0z" /></IconWrapper>;
}
function ChecklistIcon() {
  return <IconWrapper><path d="M9 7h10" /><path d="M9 12h10" /><path d="M9 17h10" /><path d="M4.5 7.5l1.2 1.2 2-2" /><path d="M4.5 12.5l1.2 1.2 2-2" /><path d="M4.5 17.5l1.2 1.2 2-2" /></IconWrapper>;
}
function BulbIcon() {
  return <IconWrapper><path d="M8 14a5 5 0 1 1 8 0c-.7.8-1.2 1.6-1.4 2.5H9.4C9.2 15.6 8.7 14.8 8 14z" /><path d="M9 19h6" /><path d="M10 21h4" /></IconWrapper>;
}
function WaveIcon() {
  return <IconWrapper><path d="M4 10v4" /><path d="M8 6v12" /><path d="M12 3v18" /><path d="M16 6v12" /><path d="M20 10v4" /></IconWrapper>;
}
function MoonIcon() {
  return <IconWrapper><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" /><line x1="16" y1="4" x2="17" y2="4" /><line x1="20" y1="4" x2="21" y2="4" /><line x1="18" y1="2" x2="18" y2="3" /><line x1="18" y1="5" x2="18" y2="6" /></IconWrapper>;
}
function CardsIcon() {
  return <IconWrapper><rect x="4" y="9" width="12" height="10" rx="2" transform="rotate(-15 4 9)" /><rect x="9" y="8" width="12" height="10" rx="2" /></IconWrapper>;
}
function HighlighterIcon() {
  return <IconWrapper><path d="M9 11l-6 6v3h3l6-6" /><path d="M11 9l4-4a2 2 0 0 1 2.8 2.8l-4 4" /><path d="M15 11l-4-4" /></IconWrapper>;
}
function UploadIcon() {
  return <IconWrapper><path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" /><path d="M12 12v9" /><path d="m8 16 4-4 4 4" /></IconWrapper>;
}
function PageIcon() {
  return <IconWrapper><path d="M7 3h8l4 4v14H7z" /><path d="M15 3v5h5" /></IconWrapper>;
}
function CopyIcon() {
  return <IconWrapper><rect x="9" y="9" width="10" height="10" rx="2" /><rect x="5" y="5" width="10" height="10" rx="2" /></IconWrapper>;
}
function GithubIcon() {
  return <IconWrapper><path d="M9 19c-4 1.2-4-2-6-2" /><path d="M15 21v-3.5c0-1 .1-1.7-.5-2.5 2.6-.3 5.5-1.3 5.5-6a4.6 4.6 0 0 0-1.3-3.2 4.3 4.3 0 0 0-.1-3.2s-1-.3-3.3 1.2a11.3 11.3 0 0 0-6 0C7 2.3 6 2.6 6 2.6a4.3 4.3 0 0 0-.1 3.2A4.6 4.6 0 0 0 4.6 9c0 4.7 2.9 5.7 5.5 6-.6.8-.8 1.5-.8 3V21" /></IconWrapper>;
}

const iconMap = {
  ask: ChatIcon,
  revision: BookOpenIcon,
  quiz: ChecklistIcon,
  explain: BulbIcon,
  audio: WaveIcon,
  night: MoonIcon,
  flashcards: CardsIcon,
  highlights: HighlighterIcon,
  upload: UploadIcon,
};

// Small controlled input wrapper for `topic` that repairs focus
// if the input is blurred immediately after a keystroke (platform glitch).
function TopicInput({ value, onChange, ...props }) {
  const ref = useRef(null);
  const lastTyped = useRef(0);

  const handleChange = (e) => {
    lastTyped.current = Date.now();
    onChange(e.target.value);
  };

  const handleBlur = () => {
    // If blur happened within 300ms of a keystroke, refocus to keep cursor
    if (Date.now() - lastTyped.current < 300) {
      ref.current?.focus();
    }
  };

  return <input ref={ref} value={value} onChange={handleChange} onBlur={handleBlur} {...props} />;
}

function App() {
  const [active, setActive] = useState('ask');

  const [mobileTab, setMobileTab] = useState('studio');
  const [modeState, setModeState] = useState(initialModeState);
  const [uploadFiles, setUploadFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(null); // { current, total, filename }
  const [uploadStatus, setUploadStatus] = useState(null);
  const [sources, setSources] = useState([]);
  const [health, setHealth] = useState(null);
  const [output, setOutput] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingLabel, setLoadingLabel] = useState('Processing…');
  const [error, setError] = useState('');
  const [uploadDragActive, setUploadDragActive] = useState(false);
  const [copiedHint, setCopiedHint] = useState('');
  const [quizAnswers, setQuizAnswers] = useState({});
  const [flashcardIndex, setFlashcardIndex] = useState(0);
  const [flashcardFlipped, setFlashcardFlipped] = useState(false);

  const uploadedCollectionHint = useMemo(() => {
    return uploadStatus?.collection_id || modeState.collectionId || 'lecture-2-notes';
  }, [modeState.collectionId, uploadStatus]);

  const updateField = (field, value) => {
    setModeState((current) => ({ ...current, [field]: value }));
  };

  const request = async (path, options = {}) => {
    const response = await fetch(apiUrl(path), options);
    const contentType = response.headers.get('content-type') || '';

    if (!response.ok) {
      let message = `Request failed with status ${response.status}`;
      if (contentType.includes('application/json')) {
        const body = await response.json();
        message = body?.detail || message;
      } else {
        message = await response.text();
      }
      throw new Error(message);
    }

    if (contentType.includes('application/json')) {
      return response.json();
    }

    return response;
  };

  const handleUpload = async (event) => {
    event.preventDefault();
    if (!uploadFiles || uploadFiles.length === 0) {
      setError('Choose at least one PDF first.');
      return;
    }
    
    setLoadingLabel('Uploading & indexing files…');
    setLoading(true);
    setError('');
    setOutput(null);

    let totalChunks = 0;
    try {
      for (let i = 0; i < uploadFiles.length; i++) {
        const file = uploadFiles[i];
        setUploadProgress({ current: i + 1, total: uploadFiles.length, filename: file.name });
        const result = await performUpload(file);
        if (result) {
          totalChunks += result.chunks_created;
        }
      }
      setUploadFiles([]);
      setUploadDragActive(false);
      setUploadProgress(null);
      setOutput({
        type: 'upload',
        title: 'Uploads complete',
        body: `Successfully ingested ${uploadFiles.length} file(s) with a total of ${totalChunks} chunks.`,
      });
      setActive('ask');
      setMobileTab('output');
    } catch (err) {
      setError(err.message);
      setUploadProgress(null);
    } finally {
      setLoading(false);
    }
  };

  const performUpload = async (file) => {
    const formData = new FormData();
    formData.append('pdf', file);

    const result = await request('/upload', {
      method: 'POST',
      body: formData,
    });
    setUploadStatus(result);
    setSources((current) => {
      const next = current.filter((item) => item.filename !== file.name);
      next.unshift({
        collection_id: result.collection_id,
        filename: file.name || 'uploaded.pdf',
        chunks_created: result.chunks_created,
      });
      return next;
    });
    return result;
  };

  const handleAsk = async (event) => {
    event.preventDefault();
    setLoadingLabel('Searching notes & generating answer…');
    setLoading(true);
    setError('');
    setOutput(null);

    try {
      const result = await request('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: modeState.question,
          language: modeState.language || 'en',
        }),
      });
      setOutput({ type: 'ask', title: 'Answer', body: result.answer, meta: result });
      setMobileTab('output');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleHealth = async () => {
    try {
      const result = await request('/health');
      setHealth(result);
    } catch (err) {
      console.error('Health check failed', err);
    }
  };

  const runJsonMode = async (path, payload, title, actionLabel) => {
    setLoadingLabel(actionLabel || `${title}…`);
    setLoading(true);
    setError('');
    setOutput(null);
    try {
      const result = await request(`/${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      setOutput({ type: path, title, body: result, meta: result });
      setMobileTab('output');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleAudio = async (event) => {
    event.preventDefault();
    setLoadingLabel('Creating audio notes — this may take 30s…');
    setLoading(true);
    setError('');
    setOutput(null);

    try {
      const response = await fetch(apiUrl('/audio-notes'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: modeState.topic,
          collection_id: modeState.collectionId || uploadedCollectionHint,
        }),
      });

      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed with status ${response.status}`);
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setOutput({
        type: 'audio',
        title: 'Audio generated',
        body: 'Your MP3 revision notes are ready.',
        meta: {
          downloadUrl: url,
          filename: `${(modeState.topic || 'audio-notes').replace(/\s+/g, '-')}.mp3`,
        },
      });
      setMobileTab('output');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!output || !output.meta) {
      setQuizAnswers({});
      setFlashcardIndex(0);
      setFlashcardFlipped(false);
      return;
    }
    if (output.type !== 'quiz') {
      setQuizAnswers({});
    }
    if (output.type !== 'flashcards') {
      setFlashcardIndex(0);
      setFlashcardFlipped(false);
    }
  }, [output]);

  useEffect(() => {
    const timer = setInterval(() => {
      handleHealth();
    }, 30000);
    handleHealth();
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const cards = output?.meta?.flashcards || [];
    const handler = (e) => {
      // ignore typing into inputs
      const tag = e.target?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA') return;

      if (e.code === 'Space' || e.key === ' ' || e.key === 'Spacebar' || e.key === 'Enter') {
        e.preventDefault();
        setFlashcardFlipped((state) => !state);
        return;
      }

      if (e.key === 'ArrowRight' || e.key === 'n') {
        if (!cards.length) return;
        setFlashcardIndex((s) => Math.min(cards.length - 1, s + 1));
        setFlashcardFlipped(false);
        return;
      }

      if (e.key === 'ArrowLeft' || e.key === 'p') {
        if (!cards.length) return;
        setFlashcardIndex((s) => Math.max(0, s - 1));
        setFlashcardFlipped(false);
        return;
      }
    };

    if (!cards.length) return;
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [output, setFlashcardFlipped, setFlashcardIndex]);

  const copyText = async (text, label) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedHint(label);
      window.setTimeout(() => setCopiedHint(''), 1400);
    } catch (_err) {
      setCopiedHint('copy-failed');
      window.setTimeout(() => setCopiedHint(''), 1400);
    }
  };

  const onDropUpload = (event) => {
    event.preventDefault();
    setUploadDragActive(false);
    const dropped = event.dataTransfer?.files?.[0];
    if (dropped && dropped.type === 'application/pdf') {
      setUploadFile(dropped);
      setActive('upload');
      setMobileTab('studio');
    }
  };

  const selectMode = (modeId) => {
    setActive(modeId);
    setMobileTab('studio');
  };

  const renderStudyForm = () => {
    switch (active) {
      case 'ask':
        return (
          <form className="mode-form" onSubmit={handleAsk}>
            <label className="field-wrap" data-filled={true}>
              <textarea rows="2" value={modeState.question} onChange={(event) => updateField('question', event.target.value)} placeholder="e.g. Explain paging in operating systems with example" style={{ resize: 'none' }} />
              <span>Your question</span>
            </label>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
              <span style={{ fontSize: '13px', color: 'var(--text-muted, #888)', fontWeight: 500 }}>Response language:</span>
              <div style={{ display: 'flex', background: 'var(--surface-2, #1e1e2e)', borderRadius: '999px', padding: '3px', gap: '3px', border: '1px solid var(--border, #333)' }}>
                {[{ code: 'en', label: '🇬🇧 English' }, { code: 'ja', label: '🇯🇵 日本語' }].map(lang => (
                  <button
                    key={lang.code}
                    type="button"
                    onClick={() => updateField('language', lang.code)}
                    style={{
                      padding: '5px 14px',
                      borderRadius: '999px',
                      border: 'none',
                      cursor: 'pointer',
                      fontSize: '13px',
                      fontWeight: 600,
                      transition: 'all 0.2s ease',
                      background: modeState.language === lang.code ? 'var(--accent, #6c63ff)' : 'transparent',
                      color: modeState.language === lang.code ? '#fff' : 'var(--text-muted, #888)',
                      boxShadow: modeState.language === lang.code ? '0 2px 8px rgba(108,99,255,0.4)' : 'none',
                    }}
                  >
                    {lang.label}
                  </button>
                ))}
              </div>
            </div>
            <button className="primary" type="submit" disabled={loading}>
              <span style={{ fontSize: '16px', marginRight: '6px' }}>✨</span> Ask Question
            </button>
          </form>
        );
      case 'upload':
        return (
          <form className="mode-form" onSubmit={handleUpload}>
            <label className="field-wrap" data-filled={uploadFiles.length > 0}>
              <input type="file" multiple accept="application/pdf" onChange={(event) => setUploadFiles(Array.from(event.target.files || []))} />
              <span>PDF Files</span>
            </label>
            <button className="primary" type="submit" disabled={loading}>Upload and index</button>
            {uploadStatus ? <p className="helper success">Indexed {uploadStatus.chunks_created} chunks.</p> : null}
          </form>
        );
      case 'revision':
        return (
          <form className="mode-form" onSubmit={(event) => {
            event.preventDefault();
            runJsonMode('revision', { topic: modeState.topic }, 'Revision sheet', 'Generating revision sheet…');
          }}>
            <label className="field-wrap" data-filled={Boolean(modeState.topic)}>
              <TopicInput value={modeState.topic} onChange={(v) => updateField('topic', v)} placeholder=" " />
              <span>Topic</span>
            </label>
            <button className="primary" type="submit" disabled={loading}>Generate revision sheet</button>
          </form>
        );
      case 'quiz':
        return (
          <form className="mode-form" onSubmit={(event) => {
            event.preventDefault();
            runJsonMode('quiz', {
              topic: modeState.topic,
              num_questions: Number(modeState.numQuestions) || 5,
            }, 'Quiz', 'Generating quiz questions…');
          }}>
            <label className="field-wrap" data-filled={Boolean(modeState.topic)}>
              <TopicInput value={modeState.topic} onChange={(v) => updateField('topic', v)} placeholder=" " />
              <span>Topic</span>
            </label>
            <label className="field-wrap" data-filled={Boolean(modeState.numQuestions)}>
              <input type="number" min="1" max="10" value={modeState.numQuestions} onChange={(event) => updateField('numQuestions', event.target.value)} placeholder=" " />
              <span>Number of questions</span>
            </label>
            <button className="primary" type="submit" disabled={loading}>Generate quiz</button>
          </form>
        );
      case 'explain':
        return (
          <form className="mode-form" onSubmit={(event) => {
            event.preventDefault();
            runJsonMode('explain-simple', { concept: modeState.concept }, 'Simple explanation', 'Explaining concept…');
          }}>
            <label className="field-wrap" data-filled={Boolean(modeState.concept)}>
              <input value={modeState.concept} onChange={(event) => updateField('concept', event.target.value)} placeholder=" " />
              <span>Concept</span>
            </label>
            <button className="primary" type="submit" disabled={loading}>Explain simply</button>
          </form>
        );
      case 'audio':
        return (
          <form className="mode-form" onSubmit={handleAudio}>
            <label className="field-wrap" data-filled={Boolean(modeState.topic)}>
              <TopicInput value={modeState.topic} onChange={(v) => updateField('topic', v)} placeholder=" " />
              <span>Topic</span>
            </label>
            <button className="primary" type="submit" disabled={loading}>Create audio notes</button>
          </form>
        );
      case 'night':
        return (
          <form className="mode-form" onSubmit={(event) => {
            event.preventDefault();
            runJsonMode('night-before', {
              subject: modeState.subject,
              exam_hours_away: Number(modeState.examHoursAway) || 8,
            }, 'Night-before sheet', 'Building your cram sheet…');
          }}>
            <label className="field-wrap" data-filled={Boolean(modeState.subject)}>
              <input value={modeState.subject} onChange={(event) => updateField('subject', event.target.value)} placeholder=" " />
              <span>Subject</span>
            </label>
            <label className="field-wrap" data-filled={Boolean(modeState.examHoursAway)}>
              <input type="number" min="1" max="72" value={modeState.examHoursAway} onChange={(event) => updateField('examHoursAway', event.target.value)} placeholder=" " />
              <span>Exam hours away</span>
            </label>
            <button className="primary" type="submit" disabled={loading}>Generate cheat sheet</button>
          </form>
        );
      case 'flashcards':
        return (
          <form className="mode-form" onSubmit={(event) => {
            event.preventDefault();
            runJsonMode('flashcards', {
              topic: modeState.topic,
              num_cards: Number(modeState.numCards) || 10,
            }, 'Flashcards', 'Generating flashcards…');
          }}>
            <label className="field-wrap" data-filled={Boolean(modeState.topic)}>
              <TopicInput value={modeState.topic} onChange={(v) => updateField('topic', v)} placeholder=" " />
              <span>Topic</span>
            </label>
            <label className="field-wrap" data-filled={Boolean(modeState.numCards)}>
              <input type="number" min="1" max="20" value={modeState.numCards} onChange={(event) => updateField('numCards', event.target.value)} placeholder=" " />
              <span>Number of cards</span>
            </label>
            <button className="primary" type="submit" disabled={loading}>Generate flashcards</button>
          </form>
        );
      case 'highlights':
        return (
          <form className="mode-form" onSubmit={(event) => {
            event.preventDefault();
            runJsonMode('highlights', { topic: modeState.topic }, 'Highlights', 'Scoring highlights…');
          }}>
            <label className="field-wrap" data-filled={Boolean(modeState.topic)}>
              <input value={modeState.topic} onChange={(event) => updateField('topic', event.target.value)} placeholder=" " />
              <span>Topic</span>
            </label>
            <button className="primary" type="submit" disabled={loading}>Score highlights</button>
          </form>
        );
      default:
        return null;
    }
  };

  const renderQuizOutput = () => {
    const quiz = output?.meta?.quiz || [];
    if (!quiz.length) return null;
    const answered = Object.keys(quizAnswers).length;
    const score = quiz.reduce((count, item, idx) => {
      const selected = quizAnswers[idx];
      return selected && selected === item.correct ? count + 1 : count;
    }, 0);

    const handleSelectOption = (qIdx, option) => {
      // prevent re-answering
      if (quizAnswers[qIdx]) return;
      setQuizAnswers((current) => ({ ...current, [qIdx]: option }));
    };

    const resetQuiz = () => {
      setQuizAnswers({});
    };

    return (
      <div className="quiz-stack">
        {quiz.map((item, idx) => {
          const selected = quizAnswers[idx];
          return (
            <article className="quiz-card" key={`${item.question}-${idx}`}>
              <h4>{idx + 1}. {item.question}</h4>
              <div className="quiz-options">
                {(item.options || []).map((option, optionIndex) => {
                  const isCorrect = option === item.correct;
                  const isSelected = selected === option;
                  const state = selected
                    ? (isCorrect ? 'correct' : (isSelected ? 'incorrect' : 'neutral'))
                    : 'default';
                  return (
                    <button
                      key={`${option}-${optionIndex}`}
                      type="button"
                      className={`quiz-option ${state}`}
                      onClick={() => handleSelectOption(idx, option)}
                      disabled={Boolean(selected)}
                    >
                      <span className="option-index">{String.fromCharCode(65 + optionIndex)}</span>
                      <span>{option}</span>
                    </button>
                  );
                })}
              </div>
              {selected ? <p className="quiz-explanation"><strong>Explanation:</strong> {item.explanation}</p> : null}
            </article>
          );
        })}

        <div className="quiz-footer">
          <p className="score-pill">{score} / {quiz.length} correct</p>
          {answered === quiz.length ? (
            <button type="button" className="primary" onClick={resetQuiz}>Retake Quiz</button>
          ) : null}
        </div>
      </div>
    );
  };

  const renderFlashcardsOutput = () => {
    const cards = output?.meta?.flashcards || [];
    if (!cards.length) return null;
    const current = cards[flashcardIndex] || cards[0];
    const difficulty = String(current.difficulty || '').toLowerCase();

    return (
      <div className="flashcards-wrap">
        <button
          type="button"
          className={`flashcard ${flashcardFlipped ? 'flipped' : ''}`}
          onClick={() => setFlashcardFlipped((s) => !s)}
          aria-label={flashcardFlipped ? 'Show question' : 'Show answer'}
        >
          <div className="card-inner">
            <div className="card-front">
              <p className="card-label">Question</p>
              <h4>{current.front}</h4>
              <span className="flip-hint">Tap to reveal answer</span>
            </div>
            <div className="card-back">
              <p className="card-label">Answer</p>
              <h4>{current.back}</h4>
              <span className={`difficulty ${difficulty}`}>{current.difficulty}</span>
            </div>
          </div>
        </button>

        <div className="flashcard-nav">
          <button type="button" className="ghost" onClick={() => {
            setFlashcardIndex((s) => Math.max(0, s - 1));
            setFlashcardFlipped(false);
          }} disabled={flashcardIndex === 0}>← Prev</button>

          <div className="card-progress">
            {cards.map((_, i) => (
              <span
                key={i}
                className={`card-progress-dot ${i === flashcardIndex ? 'active' : ''}`}
                onClick={() => { setFlashcardIndex(i); setFlashcardFlipped(false); }}
              />
            ))}
          </div>

          <button type="button" className="ghost" onClick={() => {
            setFlashcardIndex((s) => Math.min(cards.length - 1, s + 1));
            setFlashcardFlipped(false);
          }} disabled={flashcardIndex === cards.length - 1}>Next →</button>
        </div>
        <p className="mono" style={{ textAlign: 'center', fontSize: '12px', color: 'var(--text-muted)', margin: 0 }}>
          Card {flashcardIndex + 1} of {cards.length} · Space to flip · ← → to navigate
        </p>
      </div>
    );
  };


  const renderHighlightsOutput = () => {
    const highlights = output?.meta?.highlights || [];
    if (!highlights.length) return null;

    return (
      <ol className="highlights-list">
        {highlights.map((item, idx) => {
          const score = Number(item.score || 0);
          let band = 'low';
          if (score >= 8) band = 'high';
          else if (score >= 6) band = 'mid';

          return (
            <li className={`highlight-item ${band}`} key={`${item.sentence}-${idx}`}>
              <p>{item.sentence}</p>
              <div className="highlight-meta">
                <span>{item.reason}</span>
                <span className="mono">Page {item.page_number}</span>
              </div>
            </li>
          );
        })}
      </ol>
    );
  };

  const renderAudioOutput = () => {
    const downloadUrl = output?.meta?.downloadUrl;
    const filename = output?.meta?.filename;
    if (!downloadUrl) return null;

    return (
      <div className="audio-shell">
        <p className="audio-topic">{modeState.topic || 'Audio Revision Notes'}</p>
        <div className="wave-bars" aria-hidden="true">
          {Array.from({ length: 28 }).map((_, idx) => (
            <span key={idx} style={{ animationDelay: `${idx * 55}ms` }} />
          ))}
        </div>
        <audio controls src={downloadUrl} className="native-audio" />
        <a className="primary" href={downloadUrl} download={filename}>Download MP3</a>
      </div>
    );
  };



  const renderRevisionOutput = () => {
    const sheet = output?.meta?.revision_sheet;
    if (!sheet) return null;

    // Lightweight markdown → JSX: ## header, **bold**, - bullet, plain text
    const parseInline = (text) => {
      const parts = text.split(/(\*\*[^*]+\*\*)/g);
      return parts.map((part, i) =>
        part.startsWith('**') && part.endsWith('**')
          ? <strong key={i}>{part.slice(2, -2)}</strong>
          : part
      );
    };

    const lines = sheet.split('\n');
    const elements = [];
    let bulletBuffer = [];

    const flushBullets = () => {
      if (bulletBuffer.length) {
        elements.push(
          <ul className="revision-bullets" key={`ul-${elements.length}`}>
            {bulletBuffer.map((b, i) => <li key={i}>{parseInline(b)}</li>)}
          </ul>
        );
        bulletBuffer = [];
      }
    };

    lines.forEach((line, i) => {
      if (line.startsWith('## ')) {
        flushBullets();
        elements.push(<h3 className="revision-section-title" key={i}>{line.slice(3)}</h3>);
      } else if (line.startsWith('- ') || line.startsWith('* ')) {
        bulletBuffer.push(line.slice(2));
      } else if (line.trim() === '') {
        flushBullets();
      } else {
        flushBullets();
        elements.push(<p className="revision-para" key={i}>{parseInline(line)}</p>);
      }
    });
    flushBullets();

    return <div className="revision-shell">{elements}</div>;
  };

  // Shared inline markdown: **bold** → <strong>
  const parseInlineMd = (text) => {
    if (!text) return null;
    return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
      part.startsWith('**') && part.endsWith('**')
        ? <strong key={i}>{part.slice(2, -2)}</strong>
        : part
    );
  };

  // Render a markdown text block: ## headers, - bullets, plain paragraphs
  const renderMarkdownBlock = (text, className = '') => {
    if (!text) return null;
    const lines = text.split('\n');
    const els = [];
    let bullets = [];
    const flush = (k) => {
      if (bullets.length) {
        els.push(
          <ul className="revision-bullets" key={`ul-${k}`}>
            {bullets.map((b, i) => <li key={i}>{parseInlineMd(b)}</li>)}
          </ul>
        );
        bullets = [];
      }
    };
    lines.forEach((line, i) => {
      if (/^#{1,3}\s/.test(line)) {
        flush(i);
        els.push(<h3 className="revision-section-title" key={i}>{line.replace(/^#+\s/, '')}</h3>);
      } else if (/^[-*]\s/.test(line) || /^\d+\.\s/.test(line)) {
        bullets.push(line.replace(/^[-*]\s|^\d+\.\s/, ''));
      } else if (line.trim() === '') {
        flush(i);
      } else {
        flush(i);
        els.push(<p className="revision-para" key={i}>{parseInlineMd(line)}</p>);
      }
    });
    flush('end');
    return <div className={className}>{els}</div>;
  };

  const renderExplainOutput = () => {
    const meta = output?.meta || {};
    if (!meta.simple_explanation) return null;
    return (
      <div className="explain-shell">
        <div className="explain-card explain-main">
          <span className="explain-label">💡 Simple Explanation</span>
          <p>{meta.simple_explanation}</p>
        </div>
        {meta.analogy && (
          <div className="explain-card explain-analogy">
            <span className="explain-label">🔗 Analogy</span>
            <p>{meta.analogy}</p>
          </div>
        )}
        {meta.one_thing_to_remember && (
          <div className="explain-card explain-memory">
            <span className="explain-label">📌 One Thing to Remember</span>
            <p>{meta.one_thing_to_remember}</p>
          </div>
        )}
      </div>
    );
  };

  const renderNightOutput = () => {
    const meta = output?.meta || {};
    if (!meta.cheat_sheet) return null;
    return (
      <div className="night-shell">
        <h4>Exam in {modeState.examHoursAway || 8} hours — here is everything you need</h4>
        <button type="button" className="ghost" onClick={() => window.print()}>🖨 Print</button>
        {(meta.topics_covered || []).length > 0 && (
          <section>
            <h5>Top Topics</h5>
            <div className="topic-chips">
              {meta.topics_covered.map((t, i) => <span key={i} className="topic-chip">{t}</span>)}
            </div>
          </section>
        )}
        <section>
          <h5>Cram Sheet</h5>
          {renderMarkdownBlock(meta.cheat_sheet)}
        </section>
      </div>
    );
  };

  const renderOutput = () => {

    if (!output) {
      return (
        <div className="empty-state">
          <div className="empty-icon">⌁</div>
          <h3>Select a mode and run it to see results here</h3>
          <p>Try stress-ready workflows: ask, quiz, flashcards, or night-before compression.</p>
          <div className="hint-chips">
            <span>Try: Ask a question</span>
            <span>Generate flashcards</span>
            <span>Create audio</span>
          </div>
        </div>
      );
    }

    const sourcesList = output.meta?.sources || [];

    return (
      <div className="output-card">
        <h3>{output.title}</h3>
        {loading ? (
          <div className="processing-banner" role="status" aria-live="polite">
            <div className="processing-spinner" aria-hidden="true">
              <span /><span /><span /><span />
            </div>
            <div className="processing-text">
              <span className="processing-label">{loadingLabel}</span>
              <span className="processing-sub">Please wait, AI is working on this…</span>
            </div>
            <div className="skeleton-wrap" style={{ marginTop: 16 }} aria-hidden="true">
              <div className="skeleton-line full" />
              <div className="skeleton-line eighty" />
              <div className="skeleton-line sixty" />
            </div>
          </div>
        ) : null}

        {!loading && output.type === 'ask' ? renderMarkdownBlock(output.body, 'answer-md') : null}
        {!loading && (output.type === 'explain' || output.type === 'explain-simple') ? renderExplainOutput() : null}
        {!loading && output.type === 'quiz' ? renderQuizOutput() : null}
        {!loading && output.type === 'flashcards' ? renderFlashcardsOutput() : null}
        {!loading && output.type === 'highlights' ? renderHighlightsOutput() : null}
        {!loading && output.type === 'audio' ? renderAudioOutput() : null}
        {!loading && output.type === 'night-before' ? renderNightOutput() : null}
        {!loading && output.type === 'revision' ? renderRevisionOutput() : null}
        {!loading && !['ask', 'explain', 'explain-simple', 'quiz', 'flashcards', 'highlights', 'audio', 'night-before', 'revision'].includes(output.type)
          ? (typeof output.body === 'string' ? <p className="prewrap">{output.body}</p> : <pre>{JSON.stringify(output.body, null, 2)}</pre>)
          : null}

        {output.meta?.processing_time_ms ? <span className="latency">{output.meta.processing_time_ms}ms</span> : null}

        {sourcesList.length ? (
          <section className="sources-panel">
            <h4>Sources</h4>
            <div className="source-chips">
              {sourcesList.map((source, index) => (
                <button key={`${source.source_file || 'source'}-${index}`} type="button" className="source-chip">
                  <PageIcon />
                  <span>
                    {source.page_number ? `Page ${source.page_number} - ` : ''}
                    {source.source_file || 'Unknown source'}
                  </span>
                </button>
              ))}
            </div>
          </section>
        ) : null}
      </div>
    );
  };

  const Topbar = () => (
    <header className="topbar-fixed">
      <div className="brand-compact">
        <div className="brand-square">A</div>
        <span className="brand-name">AskMyNotes</span>
      </div>
      <div className="status-indicators">
        <span className={`dot ${health?.qdrant_connected ? 'green' : 'red'}`} />
        <span>Qdrant: {health?.qdrant_connected ? 'connected' : 'offline'}</span>
        <a href="https://github.com" target="_blank" rel="noreferrer" className="github-link" aria-label="GitHub"><GithubIcon /></a>
      </div>
    </header>
  );

  const SourcesPanel = () => (
    <aside className={`left-panel ${uploadDragActive ? 'drag-active' : ''}`} onDragOver={(event) => { event.preventDefault(); setUploadDragActive(true); }} onDragLeave={() => setUploadDragActive(false)} onDrop={onDropUpload}>
      <div className="panel-sticky-header">
        <h3>Sources</h3>
        <span className="count-badge">{Math.max(sources.length, uploadStatus ? 1 : 0)} PDF</span>
      </div>

        <button
          type="button"
          className={`upload-zone ${uploadDragActive ? 'drag-active over' : ''}`}
          onClick={() => document.getElementById('pdf-upload-input')?.click()}
          onDragOver={(event) => { event.preventDefault(); setUploadDragActive(true); }}
          onDragLeave={() => setUploadDragActive(false)}
          onDrop={(event) => {
            event.preventDefault();
            setUploadDragActive(false);
            const droppedFiles = Array.from(event.dataTransfer?.files || []).filter(f => f.type === 'application/pdf');
            if (droppedFiles.length > 0) {
              setUploadFiles(droppedFiles);
            }
          }}
        >
        <input
          id="pdf-upload-input"
          type="file"
          multiple
          accept="application/pdf"
          onChange={(event) => {
            setUploadFiles(Array.from(event.target.files || []));
            setActive('upload');
            setMobileTab('studio');
          }}
          hidden
        />
        <UploadIcon />
        <p>Drop PDF or click to upload</p>
      </button>

      {uploadProgress ? (
        <div className="pending-upload active-progress">
          <div>
            <strong title={uploadProgress.filename}>Indexing: {uploadProgress.filename}</strong>
            <span>Processing file {uploadProgress.current} of {uploadProgress.total}...</span>
          </div>
          <div className="progress-bar-container">
            <div className="progress-bar-fill" style={{ width: `${(uploadProgress.current / uploadProgress.total) * 100}%` }}></div>
          </div>
        </div>
      ) : uploadFiles && uploadFiles.length > 0 ? (
        <div className="pending-upload">
          <div>
            <strong>{uploadFiles.length} file(s) selected</strong>
            <span>{Math.ceil(uploadFiles.reduce((acc, f) => acc + f.size, 0) / 1024)} KB total</span>
          </div>
          <button type="button" className="primary" onClick={(event) => handleUpload(event)} disabled={loading}>Upload and index all</button>
        </div>
      ) : null}

        <div className="source-list">
          {sources.map((source, index) => (
            <article key={`${source.filename}-${index}`} className="source-card active">
              <div className="source-main">
                <span className="pdf-icon">PDF</span>
                <div>
                  <h4 title={source.filename}>{source.filename}</h4>
                  <span className="chunk-pill">
                    {source.chunks_created} chunks
                  </span>
                </div>
              </div>
            </article>
          ))}
        </div>
    </aside>
  );

  const StudioPanel = () => (
    <section className="center-panel">
      <div className="panel-sticky-header">
        <h3>Study Studio</h3>
      </div>
      <div className="mode-grid">
        {modeCards.map((item, idx) => {
          const Icon = iconMap[item.id];
          const toneClass = `tone-${String(item.tone || '').replace(/\s+/g, '-').toLowerCase()}`;
          return (
            <button type="button" key={item.id} className={`mode-card ${toneClass} ${active === item.id ? 'active' : ''}`} style={{ animationDelay: `${idx * 50}ms` }} onClick={() => selectMode(item.id)}>
              <div className="mode-top">
                <span className="mode-icon"><Icon /></span>
                <span className={`mode-dot ${item.tone}`} />
              </div>
              <div className="mode-bottom">
                <strong>{item.name}</strong>
                <p>{item.description}</p>
              </div>
            </button>
          );
        })}
      </div>

      <section className="active-form-wrap open">
        <header>
          <h4>{modeCards.find((item) => item.id === active)?.name}</h4>
          <p className="active-form-desc" style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '4px' }}>
            {modeCards.find((item) => item.id === active)?.description}
          </p>
        </header>
        {renderStudyForm()}
        {error ? (
          <div className="error-toast">
            <p>{error}</p>
            <button type="button" onClick={() => setError('')}>x</button>
          </div>
        ) : null}
      </section>
    </section>
  );

  const OutputPanel = () => (
    <section className="right-panel">
      <div className="panel-sticky-header">
        <h3>Output</h3>
        <button type="button" className="ghost" onClick={() => setOutput(null)} style={{ display: 'inline-flex', gap: '6px', alignItems: 'center' }}>
          Clear
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></svg>
        </button>
      </div>
      {output && output.type === 'ask' && modeState.question ? (
        <div className="user-query-bubble" style={{
          background: 'rgba(79, 142, 247, 0.15)',
          border: '1px solid rgba(79, 142, 247, 0.3)',
          color: 'var(--text-primary)',
          padding: '12px 16px',
          borderRadius: '12px',
          marginBottom: '16px',
          display: 'inline-block',
          marginLeft: 'auto',
          float: 'right',
          clear: 'both'
        }}>
          {modeState.question}
        </div>
      ) : null}
      <div style={{ clear: 'both' }} />
      {renderOutput()}
    </section>
  );

  const renderMobileContent = () => {
    if (mobileTab === 'sources') return SourcesPanel();
    if (mobileTab === 'studio') return StudioPanel();
    if (mobileTab === 'output') return OutputPanel();
    if (mobileTab === 'health') {
      return (
        <section className="mobile-health">
          <h3>Health</h3>
          <button type="button" className="primary" onClick={handleHealth} disabled={loading}>Check backend health</button>
          {health ? (
            <div className="health-pills">
              <span>Backend: {health.status}</span>
              <span>Qdrant: {health.qdrant_connected ? 'connected' : 'offline'}</span>
            </div>
          ) : null}
        </section>
      );
    }
    return (
      <section className="mobile-settings">
        <h3>Settings</h3>
        <p>API base: <code>{API_BASE_URL ? API_PREFIX : '/api'}</code></p>
      </section>
    );
  };

  return (
    <div className="app-root">
      <div className="desktop-shell app-shell">
        {Topbar()}
        {SourcesPanel()}
        {StudioPanel()}
        {OutputPanel()}
      </div>

      <div className="mobile-shell">
        {Topbar()}
        <div className="mobile-content">{renderMobileContent()}</div>
        <nav className="mobile-tabs">
          {[
            { id: 'sources', label: 'Sources' },
            { id: 'studio', label: 'Studio' },
            { id: 'output', label: 'Output' },
            { id: 'health', label: 'Health' },
            { id: 'settings', label: 'Settings' },
          ].map((tab) => (
            <button key={tab.id} type="button" className={mobileTab === tab.id ? 'active' : ''} onClick={() => setMobileTab(tab.id)}>
              <span>{mobileTab === tab.id ? tab.label : tab.label.slice(0, 1)}</span>
            </button>
          ))}
        </nav>
      </div>
    </div>
  );
}

export default App;
