#!/usr/bin/env bash

set -e

SESSION_NAME="camera-check"
WORKSPACE_DIR="$HOME/final_project_ws"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
	tmux attach -t "$SESSION_NAME"
	exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -c "$WORKSPACE_DIR"

# Create a 3-pane layout: one left, two stacked on the right.
tmux split-window -h -c "$WORKSPACE_DIR"
tmux select-pane -t 1
tmux split-window -v -c "$WORKSPACE_DIR"

# Left: build terminal.
tmux send-keys -t 0 "cd $WORKSPACE_DIR" C-m
tmux send-keys -t 0 "source ./vision_venv/bin/activate" C-m
tmux send-keys -t 0 "colcon build" C-m

# Top right: depth camera publisher.
tmux send-keys -t 1 "cd $WORKSPACE_DIR" C-m
tmux send-keys -t 1 "source ./vision_venv/bin/activate" C-m
tmux send-keys -t 1 "source install/setup.bash" C-m
tmux send-keys -t 1 "ros2 run depth_camera intel_pub" C-m

# Bottom right: depth camera subscriber.
tmux send-keys -t 2 "cd $WORKSPACE_DIR" C-m
tmux send-keys -t 2 "source ./vision_venv/bin/activate" C-m
tmux send-keys -t 2 "source install/setup.bash" C-m
tmux send-keys -t 2 "ros2 run depth_camera intel_sub" C-m

tmux select-pane -t 0
tmux attach -t "$SESSION_NAME"
