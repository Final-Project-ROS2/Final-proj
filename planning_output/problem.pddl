(define (problem move-block-d)
    (:domain blocksworld)
    (:objects
        a b c d - block
        table - surface
    )
    (:init
        (on-table a)
        (on-table b)
        (on-table c)
        (on d c)
        (clear a)
        (clear b)
        (clear d)
        (handempty)
    )    (:goal
        (on d b)
    )
)