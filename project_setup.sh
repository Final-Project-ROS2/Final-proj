#!/usr/bin/env bash

set -e

SESSION_NAME="final-project"
WORKSPACE_DIR="$HOME/final_project_ws"
TARGET_SUBDIR="$1"
TARGET_DIR="$WORKSPACE_DIR/src"

if [[ -n "$TARGET_SUBDIR" && -d "$WORKSPACE_DIR/src/$TARGET_SUBDIR" ]]; then
	TARGET_DIR="$WORKSPACE_DIR/src/$TARGET_SUBDIR"
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
	tmux attach -t "$SESSION_NAME"
	exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -c "$WORKSPACE_DIR"

# Create a 2x2 grid.
tmux split-window -h -c "$WORKSPACE_DIR"
tmux select-pane -t 0
tmux split-window -v -c "$WORKSPACE_DIR"
tmux select-pane -t 2
tmux split-window -v -c "$WORKSPACE_DIR"

# Function to cd to WORKSPACE_DIR and activate vision_venv
default_setup() {
	tmux send-keys -t $1 "cd $WORKSPACE_DIR" C-m
	tmux send-keys -t $1 "source ./vision_venv/bin/activate" C-m
}

# Top left: run project.
default_setup 0
tmux send-keys -t 0 "clear" C-m

# Top right: call ros2 action.
default_setup 2
tmux send-keys -t 2 "clear" C-m

# Bottom left: colcon build.
default_setup 1
tmux send-keys -t 1 "clear" C-m

# Bottom right: access code.
default_setup 3
tmux send-keys -t 3 "cd $TARGET_DIR" C-m
tmux send-keys -t 3 "clear" C-m

tmux select-pane -t 3
tmux attach -t "$SESSION_NAME"
