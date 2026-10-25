package com.eignex.koblas.consumer

import kotlin.test.Test

class ConsumerCheckTest {
    @Test
    fun `published portable products agree with reference`() {
        ConsumerCheck.run()
    }
}
