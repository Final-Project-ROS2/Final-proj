(define (domain blocksworld)
    (:requirements :strips :typing)
    (:types
        block surface
    )
    (:predicates
        (on ?b1 - block ?b2 - block)
        (on-table ?b - block)
        (clear ?b - block)
        (handempty)
        (holding ?b - block)
    )

    (:action pick-up-from-table
        :parameters (?b - block ?s - surface)
        :precondition (and (clear ?b) (on-table ?b) (handempty))
        :effect (and (not (on-table ?b)) (not (clear ?b)) (not (handempty)) (holding ?b))
    )

    (:action pick-up-from-block
        :parameters (?b - block ?b_under - block)
        :precondition (and (clear ?b) (on ?b ?b_under) (handempty))
        :effect (and (not (on ?b ?b_under)) (not (clear ?b)) (not (handempty)) (holding ?b) (clear ?b_under))
    )

    (:action put-down-on-table
        :parameters (?b - block ?s - surface)
        :precondition (holding ?b)
        :effect (and (not (holding ?b)) (handempty) (on-table ?b) (clear ?b))
    )

    (:action stack-on-block
        :parameters (?b - block ?b_under - block)
        :precondition (and (holding ?b) (clear ?b_under))
        :effect (and (not (holding ?b)) (handempty) (on ?b ?b_under) (not (clear ?b_under)) (clear ?b))
    )
)