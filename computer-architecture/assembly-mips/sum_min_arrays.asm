addi $sp, $sp, -8		# Allocate 8 bytes of stack space (for $s0 and $s1) 
sw $s0, 4($sp)			# Save $s0 on the stack 
sw $s1, 0($sp) 			# Save $s1 on the stack 
lw $s0, 0($a0) 			# Load x[0] into $s0 (initialize min1) 
lw $s1, 0($a1) 			# Load y[0] into $s1 (initialize min2) 
add $t3, $zero, $zero 	# i = 0 (initialize loop counter) 
subi $a2, $a2, 1 		# n = n - 1 (adjust loop bound to n-1) 
LOOP:
slt $t4, $t3, $a2 		# Check if i < n - 1 
beq $t4, $zero, END 	# If not, exit the loop 
addi $a0, $a0, 4 		# x++ (move to the next element in      array x) 
addi $a1, $a1, 4 		# y++ (move to the next element in array y) 
lw $t2, 0($a0) 			# Load *x into $t2 
lw $t5, 0($a1)			# Load *y into $t5 # Check if *x < min1 
slt $t4, $t2, $s0 		# Compare *x with min1 (stored in $s0) 
beq $t4, $zero, CHECK2	# If not, skip updating min1 
add $s0, $t2, $zero 	# Update min1 = *x # Check if *y < min2 
CHECK2: 
slt $t4, $t5, $s1 		# Compare *y with min2 (stored in $s1) 
beq $t4, $zero, NEXT 	# If not, skip updating min2 
add $s1, $t5, $zero 	# Update min2 = *y # Increment loop counter 
NEXT:
addi $t3, $t3, 1		# i++ 
j LOOP 					# Jump back to the start of the loop 
END: 
add $v0, $s0, $s1 		# Compute the result (min1 + min2) and store in $v0 
lw $s1, 0($sp) 			# Restore $s1 from the stack 
lw $s0, 4($sp) 			# Restore $s0 from the stack 
addi $sp, $sp, 8 		# Deallocate 8 bytes of stack space 
jr $ra 					# Return to the caller
