addi $sp, $sp, -8       # Allocate stack space for $s0 and $s1
sw $s0, 4($sp)          # Save $s0 (hash)
sw $s1, 0($sp)          # Save $s1 (c)
addi $s0, $zero, 5381   # Initialize hash = 5381
move $t0, $a0           # Copy input into $t0 (input)
LOOP:
beq $t0, $zero, END     # Exit loop if input == 0
andi $s1, $t0, 0xFF     # c = input & 0xFF (extract least significant byte)
sll $t1, $s0, 4         # t1 = hash << 4 (shift left by 4)
add $t1, $t1, $s0       # t1 = (hash << 4) + hash (hash * 17)
add $s0, $t1, $s1       # hash = ((hash << 4) + hash) + c
srl $t0, $t0, 8         # input = input >> 8 (discard least significant byte)
j LOOP                  # Repeat for the next byte
END:
move $v0, $s0           # Return the final hash value in $v0
lw $s1, 0($sp)          # Restore $s1
lw $s0, 4($sp)          # Restore $s0
addi $sp, $sp, 8        # Deallocate stack space
jr $ra                  # Return to caller
