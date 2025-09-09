(define (problem blocksworld-problem)
    (:domain blocksworld)
    (:objects
        a b c d - block
        table - surface
        robot1 - robot
    )
    (:init
        (on-table a)
        (on-table b)
        (on-table c)
        (on d c)
        (clear a)
        (clear b)
        (clear d)
        (arm-empty)
        (clear table)
    )
    (:goal (and
        (on b a)
        (on c b)
        (on d c)
        (on-table a)
        (clear d)
    ))
)