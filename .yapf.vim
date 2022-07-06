function! yapf#YAPF() range
  " Determine range to format.
  let l:cmd = 'yapf'

  " Call YAPF with the current buffer
  let l:formatted_text = systemlist(l:cmd, join(getline(1, '$'), "\n") . "\n")

  if v:shell_error
    echohl ErrorMsg
    echomsg printf('"%s" returned error: %s', l:cmd, l:formatted_text[-1])
    echohl None
    return
  endif

  " Update the buffer.
  execute '1,' . string(line('$')) . 'delete'
  call setline(1, l:formatted_text)

  " Reset cursor to first line of the formatted range.
  call cursor(a:firstline, 1)
endfunction
