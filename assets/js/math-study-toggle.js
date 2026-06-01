// Math Study View Toggle
// Allows switching between "Topics" (category-based) and "Curriculum" (timeline-based) views

const MATH_STUDY_TOGGLE_KEY = 'mathStudyViewMode';

function initMathStudyToggle() {
  const toggleBtn = document.getElementById('math-study-toggle');
  const topicsView = document.getElementById('math-study-topics-view');
  const curriculumView = document.getElementById('math-study-curriculum-view');
  const topicsLabel = document.getElementById('math-study-toggle-topics');
  const curriculumLabel = document.getElementById('math-study-toggle-curriculum');

  if (!toggleBtn || !topicsView || !curriculumView) {
    return; // Elements not found, skip initialization
  }

  // Get saved preference or default to 'topics'
  const savedMode = localStorage.getItem(MATH_STUDY_TOGGLE_KEY) || 'topics';
  setMathStudyViewMode(savedMode);

  // Toggle on button click
  toggleBtn.addEventListener('click', function () {
    const currentMode = localStorage.getItem(MATH_STUDY_TOGGLE_KEY) || 'topics';
    const newMode = currentMode === 'topics' ? 'curriculum' : 'topics';
    setMathStudyViewMode(newMode);
  });

  function setMathStudyViewMode(mode) {
    localStorage.setItem(MATH_STUDY_TOGGLE_KEY, mode);

    if (mode === 'topics') {
      topicsView.classList.add('active');
      curriculumView.classList.remove('active');
      topicsLabel.classList.add('active');
      curriculumLabel.classList.remove('active');
    } else {
      topicsView.classList.remove('active');
      curriculumView.classList.add('active');
      topicsLabel.classList.remove('active');
      curriculumLabel.classList.add('active');
    }
  }
}

// Initialize on document ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initMathStudyToggle);
} else {
  initMathStudyToggle();
}
