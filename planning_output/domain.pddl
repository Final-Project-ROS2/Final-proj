(define (domain blocks-world)
    (:requirements :strips :typing)
    (:types block) ; All blocks are of type 'block'

    (:predicates
        (on ?b1 - block ?b2 - block) ; ?b1 is on top of ?b2
        (on-table ?b - block)       ; ?b is on the table
        (clear ?b - block)          ; ?b has nothing on it
        (hand-empty)                ; The gripper is empty
        (holding ?b - block)        ; The gripper is holding ?b
    )

    ;; Action to pick up a block from the table
    (:action pick-up-from-table
        :parameters (?b - block)
        :precondition (and
            (clear ?b)
            (on-table ?b)
            (hand-empty)
        )
        :effect (and
            (not (on-table ?b))
            (not (clear ?b))
            (not (hand-empty))
            (holding ?b)
        )
    )

    ;; Action to unstack a block from another block
    (:action unstack
        :parameters (?b1 - block ?b2 - block)
        :precondition (and
            (on ?b1 ?b2)
            (clear ?b1)
            (hand-empty)
        )
        :effect (and
            (not (on ?b1 ?b2))
            (not (clear ?b1))
            (clear ?b2)
            (not (hand-empty))
            (holding ?b1)
        )
    )

    ;; Action to put down a block onto the table
    (:action put-down
        :parameters (?b - block)
        :precondition (and
            (holding ?b)
        )
        :effect (and
            (not (holding ?b))
            (on-table ?b)
            (clear ?b)
            (hand-empty)
        )
    )

    ;; Action to stack a block onto another block
    (:action stack
        :parameters (?b1 - block ?b2 - block)
        :precondition (and
            (holding ?b1)
            (clear ?b2)
        )
        :effect (and
            (not (holding ?b1))
            (on ?b1 ?b2)
            (not (clear ?b2))
            (clear ?b1)
            (hand-empty)
        )
    )
)