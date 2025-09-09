(define (domain blocksworld)
    (:requirements :strips :typing)
    (:types
        block object
        surface object
        robot object
    )
    (:predicates
        (on ?b1 - block ?b2 - block)
        (on-table ?b - block)
        (clear ?o - object)
        (holding ?b - block)
        (arm-empty)
    )

    (:action pick-up
        :parameters (?b - block ?s - surface)
        :precondition (and (clear ?b) (on-table ?b) (arm-empty))
        :effect (and (not (on-table ?b)) (not (clear ?b)) (not (arm-empty)) (holding ?b))
    )

    (:action stack
        :parameters (?b1 - block ?b2 - block)
        :precondition (and (holding ?b1) (clear ?b2))
        :effect (and (not (holding ?b1)) (not (clear ?b2)) (clear ?b1) (on ?b1 ?b2) (arm-empty))
    )

    (:action unstack
        :parameters (?b1 - block ?b2 - block)
        :precondition (and (on ?b1 ?b2) (clear ?b1) (arm-empty))
        :effect (and (not (on ?b1 ?b2)) (not (clear ?b1)) (not (arm-empty)) (holding ?b1) (clear ?b2))
    )

    (:action put-down
        :parameters (?b - block ?s - surface)
        :precondition (and (holding ?b))
        :effect (and (not (holding ?b)) (on-table ?b) (clear ?b) (arm-empty))
    )
)