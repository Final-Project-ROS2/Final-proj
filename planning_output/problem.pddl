(define (problem blocks-arrangement)
    (:domain blocks-world)
    (:objects
        a b c d - block
    )
    (:init
        ;; Initial state based on image description
        (on-table a)
        (on-table b)
        (on-table c)
        (on d c) ; Block D is on Block C

        (clear a) ; Block A is clear
        (clear b) ; Block B is clear
        (clear d) ; Block D is clear (top of its stack)
        ;; (not (clear c)) is implicitly true because D is on C

        (hand-empty)
    )
    (:goal
        ;; Goal state based on task instructions: A on B on D, C on the side
        (and
            (on a b)
            (on b d)
            (on-table d) ; D must be on the table to support the stack

            (on-table c) ; C is on the side (on the table)
            (clear c)    ; C is clear (nothing on it)
            (clear a)    ; A is clear (top of the stack)
        )
    )
)