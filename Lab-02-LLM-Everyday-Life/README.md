# Lab 02 – AI in Everyday Challenges: Using an LLM as a Product-Development Partner

**Student:** HWS Sathsarani (10967109)  
**Module:** PUSL3205 – Large Language Models and Generative AI  
**Mode:** Completed individually

## Objective
The aim of this lab is to use a Large Language Model (LLM) as a product-development partner to solve a real everyday problem faced by students. The LLM supported each stage of the design process: understanding the problem, defining the target user, generating and evaluating ideas, naming the product and creating a visual prototype.

## Chosen Challenge
**Group 2 – The Distracted Student.** Students want to study, but phone notifications, social media and noise break their concentration. This challenge was selected because it is a real personal problem, which made the user research more realistic and grounded in actual experience.

## Folder Contents
| Item | Description |
|---|---|
| `README.md` | Full documentation of the development process |
| `activity-2/activity-outcomes.md` | Final product concept, features and presentation outline |
| `activity-2/summary.md` | Summary of the work carried out with the LLM and the lessons learned |
| `screenshots/` | Evidence of every prompt and LLM response |
| `prototype/` | Individual screens of the visual prototype |
| `presentation/StillHour-Lab02.pptx` | Final presentation of the StillHour concept |

## Development Process

### 1. Understanding the problem
The problem was first described to the LLM in simple words. The prompt was then rewritten using a **role, context, task and format** structure: the LLM was asked to act as a student wellbeing researcher, given details of what had already been tried, and asked for root causes, reasons why existing solutions fail, supporting statistics and three key design insights.

![Problem analysis prompt](screenshots/1-problem-prompt.png)

The analysis showed that distraction has two types of causes:
- **External:** phone notifications and small, sudden noises.
- **Internal:** the habit of checking social media, stress, mental fatigue and a lack of feedback on focus.

Research showed that simply having a phone nearby reduces available attention, even when it is switched off (Ward et al., 2017), and that fewer than 1 in 25 students can study for an hour without picking up their phone (The Boar, 2025). This explained why switching off the phone and studying in a quiet room had not fully solved the problem: both approaches only address external distractions.

![Root causes part 1](screenshots/2-root-causes-1.png)
![Root causes part 2](screenshots/3-root-causes-2.png)
![Root causes part 3](screenshots/4-root-causes-3.png)

### 2. Defining the target user
A user persona was created to represent the target user. The prompt included only real details: a 20-year-old final-year Data Science student who studies alone at home from 6:00 to 8:00 am, can focus for about 40 minutes, keeps her phone on the desk, is distracted by Instagram, TikTok and small noises, and scrolls social media before sleeping.

An earlier version of the persona contained details that had never been provided, such as an invented daily routine. The prompt was therefore rewritten with the constraint *"do not add any details that I have not mentioned"*, which produced an accurate persona named **Tharushi**.

![Persona prompt](screenshots/5-persona-prompt.png)
![Persona part 1](screenshots/6-persona-1.png)
![Persona part 2](screenshots/7-persona-2.png)
![Persona part 3](screenshots/8-persona-3.png)

### 3. Generating ideas
The LLM was asked to generate six product ideas that combined mobile apps, physical devices and hybrid products. Each idea included a short description, how it solves the persona's problems and one possible weakness, so the ideas could be compared fairly.

![Ideas prompt](screenshots/9-ideas-prompt.png)
![Six product ideas](screenshots/10-six-ideas.png)
![Most promising ideas](screenshots/11-most-promising.png)

### 4. Evaluating and refining the chosen idea
**StudyNest Hub** was selected because it was the only idea that addressed both external and internal distractions in one product. The idea was then refined with three original features:
- a phone lock with a visual countdown (hourglass, growing tree or house being built)
- a single desk device combining a focus lamp and adaptive noise masking
- a bedtime light mode that locks social media at night but still allows phone calls

The LLM evaluated the refined idea, identified its strengths and weaknesses, and suggested ways to reduce the weaknesses, such as a "commitment unlock" to stop users from ending sessions on impulse and an emergency-contact list for important calls.

![Evaluation prompt](screenshots/12-evaluation-prompt.png)
![Strengths](screenshots/13-strengths.png)
![Weaknesses and solutions](screenshots/14-weaknesses.png)

### 5. Naming the product
The LLM suggested five names and recommended "FocusNest". The name **StillHour** was chosen instead, because it reflects the calm, undisturbed early-morning study hour that the product is designed to protect. The feature lists below still show the earlier working name.

![Name options](screenshots/15-name-options.png)
![Device features](screenshots/16-device-features.png)
![App features](screenshots/17-app-features.png)
![Typical study session](screenshots/18-study-session.png)

### 6. Visual Prototype
A visual prototype was created to show how StillHour works: four mobile app screens and the StillHour Hub desk device.

![StillHour prototype overview](screenshots/20-prototype-board.png)

| Home | Focus session | Bedtime light mode | Focus insights |
|---|---|---|---|
| ![Home](prototype/1-home.png) | ![Focus session](prototype/2-focus-session.png) | ![Bedtime light mode](prototype/3-bedtime-light-mode.png) | ![Focus insights](prototype/4-focus-insights.png) |
| Choose the countdown animation and session length | Countdown ring, time left and "phone locked" | Social media locked at night, calls still allowed | Weekly focus time, streak and collection (sample data) |

![StillHour Hub](prototype/5-stillhour-hub.png)

The **StillHour Hub** combines a ring focus lamp that fills as the session counts down, an adaptive noise-masking speaker, a phone dock that locks and charges the phone, and a one-touch start button.

![Prototype summary](screenshots/19-prototype-summary.png)

## Final Product
The complete product concept, feature list and presentation outline are documented in [activity-2/activity-outcomes.md](activity-2/activity-outcomes.md).

## Observations
- **Prompt quality directly affected the quality of the output.** Giving the LLM a role, clear context, numbered tasks and a required format produced far more focused and useful answers than a simple question.
- **The LLM filled missing information with invented details.** The first persona included a routine that had never been described. Adding explicit constraints reduced this problem.
- **The LLM was strong at analysing, structuring and generating ideas quickly**, which made it a useful partner for early-stage design.
- **Human judgement was essential for the final decisions.** Choosing the idea, adding the countdown, lamp and bedtime features, and selecting the name came from personal experience rather than from the LLM.
- **Statistics provided by an LLM need to be checked** against their original sources before being used.

## Learning Outcomes
- Applied an LLM at every stage of product development, from problem research to a visual prototype.
- Practised prompt-engineering techniques: role prompting, providing context, structured tasks, output formatting and constraints.
- Recognised the limitations of LLMs, particularly invented details, and the importance of human decision-making and verification.

## References
The Boar (2025) *Most students want to reduce screen time, as study finds they risk scrolling for 25 years of their life*. Available at: https://theboar.org/2025/07/most-students-want-to-reduce-screen-time-as-study-finds-they-risk-scrolling-for-25-years-of-their-life (Accessed: 18 November 2026).

Ward, A.F., Duke, K., Gneezy, A. and Bos, M.W. (2017) 'Brain drain: the mere presence of one's own smartphone reduces available cognitive capacity', *Journal of the Association for Consumer Research*, 2(2), pp. 140–154.