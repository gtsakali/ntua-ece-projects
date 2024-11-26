.data
data_space:  .space 32           # Space for the `data` array (8 * sizeof(int) = 32 bytes)
tree_space:  .space 60           # Space for the `tree` array (15 * sizeof(int) = 60 bytes)
print_msg:   .asciiz "Root hash: %d\n"

.text
.globl main
main:
    # Allocate `data` array in memory
    la $t0, data_space          # Load address of `data_space` into $t0
    move $s0, $t0               # Save base address of `data` in $s0 for later use

    # data[0] = ('1' << 16) | ('9' << 8) | '5'
    li $t1, '1'                 # Load ASCII '1' into $t1
    sll $t1, $t1, 16            # Shift '1' by 16 bits (most significant)
    li $t2, '9'                 # Load ASCII '9' into $t2
    sll $t2, $t2, 8             # Shift '9' by 8 bits
    li $t3, '5'                 # Load ASCII '5' into $t3
    or $t1, $t1, $t2            # Combine '1' and '9'
    or $t1, $t1, $t3            # Combine with '5'
    sw $t1, 0($s0)              # Store result in data[0]

    # data[1] = ('N' << 24) | ('T' << 16) | ('U' << 8) | 'A'
    li $t1, 'N'                 # Load ASCII 'N' into $t1
    sll $t1, $t1, 24            # Shift 'N' by 24 bits
    li $t2, 'T'                 # Load ASCII 'T' into $t2
    sll $t2, $t2, 16            # Shift 'T' by 16 bits
    li $t3, 'U'                 # Load ASCII 'U' into $t3
    sll $t3, $t3, 8             # Shift 'U' by 8 bits
    li $t4, 'A'                 # Load ASCII 'A' into $t4
    or $t1, $t1, $t2            # Combine 'N' and 'T'
    or $t1, $t1, $t3            # Combine with 'U'
    or $t1, $t1, $t4            # Combine with 'A'
    sw $t1, 4($s0)              # Store result in data[1]

    # data[2] = ('E' << 16) | ('C' << 8) | 'E'
    li $t1, 'E'                 # Load ASCII 'E' into $t1
    sll $t1, $t1, 16            # Shift 'E' by 16 bits
    li $t2, 'C'                 # Load ASCII 'C' into $t2
    sll $t2, $t2, 8             # Shift 'C' by 8 bits
    li $t3, 'E'                 # Load ASCII 'E' into $t3
    or $t1, $t1, $t2            # Combine 'E' and 'C'
    or $t1, $t1, $t3            # Combine with 'E'
    sw $t1, 8($s0)              # Store result in data[2]

    # data[3] = ('C' << 8) | 'A'
    li $t1, 'C'                 # Load ASCII 'C' into $t1
    sll $t1, $t1, 8             # Shift 'C' by 8 bits
    li $t2, 'A'                 # Load ASCII 'A' into $t2
    or $t1, $t1, $t2            # Combine 'C' and 'A'
    sw $t1, 12($s0)             # Store result in data[3]

    # data[4] = ('2' << 24) | ('0' << 16) | ('2' << 8) | '4'
    li $t1, '2'                 # Load ASCII '2' into $t1
    sll $t1, $t1, 24            # Shift '2' by 24 bits
    li $t2, '0'                 # Load ASCII '0' into $t2
    sll $t2, $t2, 16            # Shift '0' by 16 bits
    li $t3, '2'                 # Load ASCII '2' into $t3
    sll $t3, $t3, 8             # Shift '2' by 8 bits
    li $t4, '4'                 # Load ASCII '4' into $t4
    or $t1, $t1, $t2            # Combine '2' and '0'
    or $t1, $t1, $t3            # Combine with '2'
    or $t1, $t1, $t4            # Combine with '4'
    sw $t1, 16($s0)             # Store result in data[4]

    # data[5] = ('0' << 24) | ('6' << 16) | ('0' << 8) | '1'
    li $t1, '0'
    sll $t1, $t1, 24
    li $t2, '6'
    sll $t2, $t2, 16
    li $t3, '0'
    sll $t3, $t3, 8
    li $t4, '1'
    or $t1, $t1, $t2
    or $t1, $t1, $t3
    or $t1, $t1, $t4
    sw $t1, 20($s0)

    # data[6] = ('2' << 24) | ('0' << 16) | ('0' << 8) | '4'
    li $t1, '2'
    sll $t1, $t1, 24
    li $t2, '0'
    sll $t2, $t2, 16
    li $t3, '0'
    sll $t3, $t3, 8
    li $t4, '4'
    or $t1, $t1, $t2
    or $t1, $t1, $t3
    or $t1, $t1, $t4
    sw $t1, 24($s0)

    # data[7] = ('M' << 24) | ('I' << 16) | ('P' << 8) | 'S'
    li $t1, 'M'
    sll $t1, $t1, 24
    li $t2, 'I'
    sll $t2, $t2, 16
    li $t3, 'P'
    sll $t3, $t3, 8
    li $t4, 'S'
    or $t1, $t1, $t2
    or $t1, $t1, $t3
    or $t1, $t1, $t4
    sw $t1, 28($s0)

    # Call create_leaves
    la $a0, data_space         # Pass data array as first argument
    li $a1, 8                  # Pass array size (8 elements)
    la $a2, tree_space         # Pass tree array as third argument
    jal create_leaves          # Call create_leaves

    # Call create_Merkle_Tree
    la $a0, tree_space         # Pass tree array as first argument
    li $a1, 8                  # Pass number of leaves (8 elements)
    jal create_Merkle_Tree     # Call create_Merkle_Tree

    # Print the root hash
    move $a0, $v0              # Move root hash (returned in $v0) to $a0
    la $a1, print_msg          # Load address of print message
    li $v0, 4                  # Syscall for print_string
    syscall

    # Exit program
    li $v0, 10                 # Syscall for exit
    syscall
