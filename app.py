from flask import Flask, render_template, request, redirect

app = Flask(__name__)

tasks = [
    {"task": "Wake up early", "time": "05:00", "done": False},
    {"task": "Morning exercise", "time": "06:00", "done": False},
    {"task": "Read newspaper", "time": "07:00", "done": False},
    {"task": "Attend college classes", "time": "09:00", "done": False},
    {"task": "Complete Python practice", "time": "11:00", "done": False},
    {"task": "Work on internship project", "time": "13:00", "done": False},
    {"task": "Lunch break", "time": "14:00", "done": False},
    {"task": "Study web development", "time": "16:00", "done": False},
    {"task": "Revise notes", "time": "19:00", "done": False},
    {"task": "Sleep early", "time": "22:00", "done": False},
]

@app.route('/')
def home():
    done = sum(1 for t in tasks if t['done'])
    return render_template('index.html', tasks=tasks, done=done, total=len(tasks))

@app.route('/add', methods=['POST'])
def add_task():
    task_name = request.form.get('task', '').strip()
    task_time = request.form.get('time', '').strip()
    if task_name and task_time:
        tasks.append({"task": task_name, "time": task_time, "done": False})
    return redirect('/')

@app.route('/toggle/<int:index>')
def toggle(index):
    if 0 <= index < len(tasks):
        tasks[index]['done'] = not tasks[index]['done']
    return redirect('/')

@app.route('/delete/<int:index>')
def delete(index):
    if 0 <= index < len(tasks):
        tasks.pop(index)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)
